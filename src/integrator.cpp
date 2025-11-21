#include "rdr/integrator.h"

#include <omp.h>

#include "rdr/bsdf.h"
#include "rdr/camera.h"
#include "rdr/canary.h"
#include "rdr/film.h"
#include "rdr/halton.h"
#include "rdr/interaction.h"
#include "rdr/light.h"
#include "rdr/math_aliases.h"
#include "rdr/math_utils.h"
#include "rdr/platform.h"
#include "rdr/properties.h"
#include "rdr/ray.h"
#include "rdr/scene.h"
#include "rdr/sdtree.h"

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 *
 * Intersection Test Integrator's Implementation
 *
 * ===================================================================== */

void IntersectionTestIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // Statistics
  std::atomic<int> cnt = 0;

  const Vec2i &resolution = camera->getFilm()->getResolution();
#pragma omp parallel for schedule(dynamic)
  for (int dx = 0; dx < resolution.x; dx++) {
    ++cnt;
    if (cnt % (resolution.x / 10) == 0)
      Info_("Rendering: {:.02f}%", cnt * 100.0 / resolution.x);
    Sampler sampler;
    for (int dy = 0; dy < resolution.y; dy++) {
      sampler.setPixelIndex2D(Vec2i(dx, dy));
      for (int sample = 0; sample < spp; sample++) {
        // 多重采样抗锯齿：为当前像素生成随机子像素样本
        const Vec2f pixel_sample = sampler.getPixelSample();
        auto ray                 = camera->generateDifferentialRay(pixel_sample.x, pixel_sample.y);
        const Vec3f L            = Li(scene, ray, sampler);
        // 调试范围断言
        assert(pixel_sample.x >= dx && pixel_sample.x <= dx + 1);
        assert(pixel_sample.y >= dy && pixel_sample.y <= dy + 1);
        camera->getFilm()->commitSample(pixel_sample, L);
      }
    }
  }
}

Vec3f IntersectionTestIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  Vec3f color(0.0);

  // Cast a ray until we hit a non-specular surface or miss
  // Record whether we have found a diffuse surface
  bool diffuse_found = false;
  SurfaceInteraction interaction;

  for (int i = 0; i < max_depth; ++i) {
    interaction      = SurfaceInteraction();
    bool intersected = scene->intersect(ray, interaction);

    // Perform RTTI to determine the type of the surface
    bool is_ideal_diffuse =
        dynamic_cast<const IdealDiffusion *>(interaction.bsdf) != nullptr;
    bool is_perfect_refraction =
        dynamic_cast<const PerfectRefraction *>(interaction.bsdf) != nullptr;

    // Set the outgoing direction
    interaction.wo = -ray.direction;

    if (!intersected) {
      break;
    }

    if (is_perfect_refraction) {
      // 跟踪完美折射路径：采样 BSDF 获得新的入射方向 wi
      Float pdf = 0.0F;
      Vec3f f   = interaction.bsdf->sample(interaction, sampler, &pdf);
      (void)f; // 本直接光照积分器不累积镜面部分，只前进路径
      // interaction.wi 已由 sample 填写；沿该方向继续前行
      ray = interaction.spawnRay(interaction.wi);
      continue;
    }

    if (is_ideal_diffuse) {
      // We only consider diffuse surfaces for direct lighting
      diffuse_found = true;
      break;
    }

    // We simply omit any other types of surfaces
    break;
  }

  if (!diffuse_found) {
    return color;
  }

  color = directLighting(scene, interaction);
  return color;
}

Vec3f IntersectionTestIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction) const {
  Vec3f color(0.0F);
  Float dist_to_light = Norm(point_light_position - interaction.p);
  Vec3f light_dir     = Normalize(point_light_position - interaction.p);

  // 阴影测试：从交点向光源发出 shadow ray，若在到达光源前被阻挡则返回黑色
  SurfaceInteraction shadow_it;
  Ray shadow_ray = interaction.spawnRayTo(point_light_position);
  bool blocked   = scene->intersect(shadow_ray, shadow_it);
  if (blocked) {
    // 如果命中的点距离交点明显小于光源距离，认为被遮挡
    Float hit_dist = Norm(shadow_it.p - interaction.p);
    if (hit_dist + 1e-4F < dist_to_light) {
      return Vec3f(0.0F);
    }
  }

  // 漫反射直接光照估计
  const BSDF *bsdf      = interaction.bsdf;
  bool is_ideal_diffuse = dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;
  if (bsdf != nullptr && is_ideal_diffuse) {
    // 简单 Lambert 模型：albedo * cos(theta) * 点光源衰减
    Float cos_theta = std::max(Dot(light_dir, interaction.normal), 0.0F);
    Vec3f albedo    = bsdf->evaluate(interaction) * cos_theta;
    // 点光源平方反比衰减（可选，保持简单也可不加）
    Float inv_r2 = 1.0F / (dist_to_light * dist_to_light + 1e-4F);
    color       += point_light_flux * albedo * inv_r2;
  }
  return color;
}

/* ===================================================================== *
 *
 * Path Integrator's Implementation
 *
 * ===================================================================== */

void PathIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  const int width  = camera->getFilm()->getResolution().x;
  const int height = camera->getFilm()->getResolution().y;
  std::atomic<int> cnt = 0;

  #pragma omp parallel for schedule(dynamic)
  for (int y = 0; y < height; ++y) {
    ++cnt;
    if (cnt % (height / 10) == 0)
      Info_("Rendering: {:.02f}%", cnt * 100.0 / height);
    Sampler sampler;
    for (int x = 0; x < width; ++x) {
      sampler.setPixelIndex2D(Vec2i(x, y));
      for (int s = 0; s < spp; ++s) {
        Vec2f pixel_sample = sampler.getPixelSample();
        auto ray = camera->generateDifferentialRay(pixel_sample.x, pixel_sample.y);
        Vec3f L = Li(scene, ray, sampler);
        camera->getFilm()->commitSample(pixel_sample, L);
      }
    }
  }
}

namespace {
Vec3f evalEmission(ref<Scene> scene, const Ray& ray, const SurfaceInteraction& interaction, bool intersected) {
    if (intersected) {
        if (interaction.isLight()) {
            return interaction.light->Le(interaction, -ray.direction);
        }
    } else {
        if (scene->hasInfiniteLight()) {
            return scene->getInfiniteLight()->Le(interaction, -ray.direction);
        }
    }
    return Vec3f(0.0F);
}
} // namespace

Vec3f PathIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  Vec3f radiance(0.0F);
  Vec3f beta(1.0F);
  Ray current_ray(ray.origin, ray.direction);
  bool specular_bounce = false;

  for (int bounces = 0; bounces < max_depth; ++bounces) {
    SurfaceInteraction interaction;
    bool intersected = scene->intersect(current_ray, interaction);

    // Handle Le (Emission)
    if (bounces == 0 || specular_bounce) {
      radiance += beta * evalEmission(scene, current_ray, interaction, intersected);
    }
    
    if (!intersected) break;

    if (interaction.bsdf == nullptr) break;

    // Direct Lighting (NEE)
    // Only if not specular. If specular, we rely on the next bounce to hit light.
    if (!interaction.bsdf->isDelta()) {
      radiance += beta * directLighting(scene, interaction, sampler);
    }

    // Sample BSDF for next direction
    Float pdf = 0.0F;
    Vec3f f = interaction.bsdf->sample(interaction, sampler, &pdf);
    
    if (f == Vec3f(0.0F) || pdf == 0.0F) break;
    
    beta *= f * std::abs(Dot(interaction.normal, interaction.wi)) / pdf;
    specular_bounce = interaction.bsdf->isDelta();
    current_ray = interaction.spawnRay(interaction.wi);
    
    // Russian Roulette
    if (bounces > 3) {
        Float q = std::max(0.05F, 1.0F - Max(beta.x, beta.y, beta.z));
        if (sampler.get1D() < q) break;
        beta /= (1 - q);
    }
  }
  return radiance;
}

Vec3f PathIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction, Sampler &sampler) {
  Vec3f direct_light(0.0F);
  const auto &lights = scene->getLights();
  if (lights.empty()) return direct_light;
  
  SurfaceInteraction light_interaction = scene->sampleEmitterDirect(interaction, sampler);
  
  Vec3f wi_world = interaction.wi; 
  Vec3f f = interaction.bsdf->evaluate(interaction);
  
  if (f == Vec3f(0.0F)) return direct_light;
  
  Ray shadow_ray;
  Float dist = 1.0F;
  
  if (light_interaction.isInfLight()) {
      shadow_ray = interaction.spawnRay(wi_world);
      dist = Float_INF;
  } else {
      shadow_ray = interaction.spawnRayTo(light_interaction.p);
      dist = Norm(light_interaction.p - interaction.p);
  }
  
  SurfaceInteraction shadow_interaction;
  bool occluded = scene->intersect(shadow_ray, shadow_interaction);
  if (occluded) {
      if (!light_interaction.isInfLight()) {
          Float hit_dist = Norm(shadow_interaction.p - interaction.p);
          if (hit_dist < dist - 1e-4F) return direct_light;
      } else {
          return direct_light;
      }
  }
  
  Vec3f emission = light_interaction.light->Le(light_interaction, -wi_world);
  if (emission == Vec3f(0.0F)) return direct_light;
  
  Float pdf = light_interaction.pdf;
  if (pdf == 0.0F) return direct_light;
  
  if (light_interaction.measure == EMeasure::EArea) {
      Float geometry_term = std::abs(Dot(interaction.normal, wi_world)) * std::abs(Dot(light_interaction.normal, -wi_world)) / (dist * dist);
      direct_light = f * emission * geometry_term / pdf;
  } else if (light_interaction.measure == EMeasure::ESolidAngle) {
      direct_light = f * emission * std::abs(Dot(interaction.normal, wi_world)) / pdf;
  }
  
  return direct_light;
}

/* ===================================================================== *
 *
 * New Integrator's Implementation
 *
 * ===================================================================== */

// Instantiate template
// clang-format off
template Vec3f
IncrementalPathIntegrator::Li<Path>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
template Vec3f
IncrementalPathIntegrator::Li<PathImmediate>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
// clang-format on

// This is exactly a way to separate dec and def
template <typename PathType>
Vec3f IncrementalPathIntegrator::Li(  // NOLINT
    [[maybe_unused]] ref<Scene> scene, [[maybe_unused]] DifferentialRay &ray, [[maybe_unused]] Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

RDR_NAMESPACE_END
