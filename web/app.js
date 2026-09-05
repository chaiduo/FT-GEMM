const heroImage = document.querySelector("#hero-image");
const stageLinks = [...document.querySelectorAll(".stage-nav a")];
const stages = [...document.querySelectorAll(".stage-section")];

function updateHeroImage() {
  if (!heroImage) return;
  const isGeneratedArtwork = heroImage.naturalWidth / heroImage.naturalHeight > 1.4;
  heroImage.classList.toggle("is-ready", isGeneratedArtwork);
}

if (heroImage) {
  heroImage.addEventListener("load", updateHeroImage);
  heroImage.addEventListener("error", () => heroImage.classList.remove("is-ready"));
  if (heroImage.complete) updateHeroImage();
}

function setCurrentStage(stage) {
  stageLinks.forEach((link) => {
    const selected = link.dataset.stage === stage;
    link.classList.toggle("is-current", selected);
    if (selected) link.setAttribute("aria-current", "step");
    else link.removeAttribute("aria-current");
  });
}

if ("IntersectionObserver" in window) {
  const stageObserver = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((left, right) => right.intersectionRatio - left.intersectionRatio)[0];
      if (visible) setCurrentStage(visible.target.dataset.stage);
    },
    { rootMargin: "-18% 0px -58% 0px", threshold: [0.1, 0.3, 0.6] },
  );
  stages.forEach((stage) => stageObserver.observe(stage));
}

stageLinks.forEach((link) => {
  link.addEventListener("click", () => setCurrentStage(link.dataset.stage));
});
