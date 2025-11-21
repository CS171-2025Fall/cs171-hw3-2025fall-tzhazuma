#include "rdr/light.h"

#include "rdr/accel.h"
#include "rdr/interaction.h"
#include "rdr/scene.h"
#include "rdr/shape.h"
#include "rdr/texture.h"

RDR_NAMESPACE_BEGIN

AreaLight::AreaLight(const Properties &props)
    : Light(props),
      radiance(props.getProperty<Vec3f>("radiance", Vec3f(1, 1, 1))) {}

void AreaLight::crossConfiguration(const CrossConfigurationContext &) {
  clearProperties();
}

Vec3f AreaLight::Le(
    const SurfaceInteraction &interaction, const Vec3f &w) const {
  assert(interaction.light == this);
  assert(interaction.type == ESurfaceInteractionType::ELight);
  return Dot(w, interaction.normal) > 0.0 ? radiance : Vec3f(0.0);
}

Float AreaLight::pdf(const SurfaceInteraction &interaction) const {
  return shape->pdf(interaction);
}

Float AreaLight::pdfDirection(
    const SurfaceInteraction &interaction, const Vec3f &w) const {
  Frame frame(interaction.normal);
  return max(Dot(frame.WorldToLocal(w), DefaultFrameLocalNormal) / PI, 0.0F);
}

SurfaceInteraction AreaLight::sample(
    SurfaceInteraction &interaction, Sampler &sampler) const {
  SurfaceInteraction light_interaction = shape->sample(sampler);

  // Further fill the interaction.
  light_interaction.type = ESurfaceInteractionType::ELight;
  light_interaction.setPrimitive(nullptr, this, nullptr);
  light_interaction.wo = Normalize(interaction.p - light_interaction.p);

  interaction.wi = -light_interaction.wo;

  return light_interaction;
}

SurfaceInteraction AreaLight::sample(Sampler &sampler) const {
  SurfaceInteraction light_interaction = shape->sample(sampler);

  light_interaction.type = ESurfaceInteractionType::ELight;
  light_interaction.setPrimitive(nullptr, this, nullptr);

  return light_interaction;
}

Vec3f AreaLight::sampleDirection(
    const SurfaceInteraction &interaction, Sampler &sampler, Float &pdf) const {
  Frame frame(interaction.normal);
#if false
  const Vec3f &local_outgoing_direction =
      UniformSampleHemisphere(sampler.get2D());
  pdf = 1 / (2 * PI);
#endif

  // Cosine-weighted
  const Vec3f &local_outgoing_direction =
      CosineSampleHemisphere(sampler.get2D());
  pdf = Dot(local_outgoing_direction, DefaultFrameLocalNormal) / PI;
  return frame.LocalToWorld(local_outgoing_direction);
}

Float AreaLight::energy() const {
  return 2 * PI * shape->area() * (radiance.x + radiance.y + radiance.z) / 3;
}

void InfiniteAreaLight::crossConfiguration(
    const CrossConfigurationContext &context) {
  auto texture_name = properties.getProperty<std::string>("texture_name");
  auto texture_ptr  = context.textures.find(texture_name);
  if (texture_ptr != context.textures.end()) {
    texture = texture_ptr->second;
  } else {
    Exception_("Texture {} not found", texture_name);
  }

  clearProperties();
}

Vec3f InfiniteAreaLight::Le(
    const SurfaceInteraction &interaction, const Vec3f &w) const {
  // interaction is not used
  // w is pointing outward
  Vec2f scoord =
      InverseSphericalDirection(dirWorldToLocal(-w));  // (theta, phi)

  // A rotation must be performed, since global and spherical coordinate
  // system are different
  Vec2f uv(scoord[1] / (2.0 * PI), scoord[0] / PI);

  auto new_interaction = interaction;
  new_interaction.setUV(uv);

  // This is infinite light, dudx = dvdx = dudy = dvdy = 0
  return texture->evaluate(new_interaction) * scale;
}

Float InfiniteAreaLight::pdf(const SurfaceInteraction &interaction) const {
  if (distribution) {
    Vec3f local_w = dirWorldToLocal(-interaction.wo);
    Vec2f scoord  = InverseSphericalDirection(local_w);
    Float theta   = scoord[0];
    Float phi     = scoord[1];
    Float u       = phi / (2 * PI);
    Float v       = theta / PI;
    Float pdf_uv  = distribution->pdf(Vec2f(u, v));
    Float sinTheta = std::sin(theta);
    if (sinTheta == 0) return 0;
    return pdf_uv / (2 * PI * PI * sinTheta);
  }
  return 1.0 / (4.0 * PI);
}

Float InfiniteAreaLight::pdfDirection(
    const SurfaceInteraction &, const Vec3f &w) const {
  if (distribution) {
    Vec3f local_w = dirWorldToLocal(-w);
    Vec2f scoord  = InverseSphericalDirection(local_w);
    Float theta   = scoord[0];
    Float phi     = scoord[1];
    Float u       = phi / (2 * PI);
    Float v       = theta / PI;
    Float pdf_uv  = distribution->pdf(Vec2f(u, v));
    Float sinTheta = std::sin(theta);
    if (sinTheta == 0) return 0;
    return pdf_uv / (2 * PI * PI * sinTheta);
  }
  return 1.0 / (4.0 * PI);
}

SurfaceInteraction InfiniteAreaLight::sample(
    SurfaceInteraction &interaction, Sampler &sampler) const {
  Vec3f w;
  Float pdf_val;
  if (distribution) {
    Vec2f uv = distribution->sampleContinuous(sampler.get2D(), &pdf_val);
    if (pdf_val == 0) return SurfaceInteraction{};
    Float phi      = uv[0] * 2 * PI;
    Float theta    = uv[1] * PI;
    Vec3f local_w  = SphericalDirection(theta, phi);
    w              = dirLocalToWorld(local_w);
    Float sinTheta = std::sin(theta);
    if (sinTheta == 0)
      pdf_val = 0;
    else
      pdf_val /= (2 * PI * PI * sinTheta);
  } else {
    w       = UniformSampleSphere(sampler.get2D());
    pdf_val = 1.0 / (4.0 * PI);
  }

  auto light_interaction = sampleFromOutgoingDirection(-w);
  light_interaction.setPdf(pdf_val, EMeasure::ESolidAngle);
  interaction.wi = w;
  return light_interaction;
}

SurfaceInteraction InfiniteAreaLight::sample(Sampler &sampler) const {
  // TODO: optimize implementation
  SurfaceInteraction interaction;
  return this->sample(interaction, sampler);
}

Vec3f InfiniteAreaLight::sampleDirection(
    const SurfaceInteraction &interaction, Sampler &, Float &pdf) const {
  // The outgoing-direction of InfiniteAreaLight is pre-determined
  pdf = 1.0;
  return interaction.normal;
}

SurfaceInteraction InfiniteAreaLight::sampleFromOutgoingDirection(
    const Vec3f &w) const {
  AssertAllNormalized(w);
  SurfaceInteraction light_interaction{};
  const auto local_w = dirWorldToLocal(w);

  light_interaction.type = ESurfaceInteractionType::EInfLight;
  light_interaction.wo   = w;

  const auto intersection_p = -local_w * radius;
  light_interaction.setGeneral(pointLocalToWorld(intersection_p) + scene_center,
      dirLocalToWorld(local_w));
  light_interaction.setPrimitive(nullptr, this, nullptr);

  return light_interaction;
}

void InfiniteAreaLight::preprocess(const PreprocessContext &context) {
  const auto &bound = context.scene->getBound();
  scene_center      = bound.getCenter();
  radius            = 2 * Max(Norm(bound.upper_bnd - scene_center),
                              Norm(bound.low_bnd - scene_center), radius);

  if (auto *img_tex = dynamic_cast<ImageTexture *>(texture.get())) {
    int width         = img_tex->getWidth();
    int height        = img_tex->getHeight();
    const Float *data = img_tex->getData();
    std::vector<Float> img(width * height);
    for (int v = 0; v < height; ++v) {
      Float vp       = (v + 0.5f) / (Float)height;
      Float sinTheta = std::sin(PI * vp);
      for (int u = 0; u < width; ++u) {
        int index = v * width + u;
        Float val = 0.2126f * data[4 * index] + 0.7152f * data[4 * index + 1] +
                    0.0722f * data[4 * index + 2];
        img[index] = val * sinTheta;
      }
    }
    distribution = make_ref<Distribution2D>(img.data(), width, height);
  }
}

RDR_NAMESPACE_END
