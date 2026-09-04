const points = [...document.querySelectorAll(".chart-point")];
const cards = [...document.querySelectorAll(".stage-card")];
const tooltip = document.querySelector("#chart-tooltip");

function selectStage(stage) {
  const point = points.find((item) => item.dataset.stage === stage);
  const card = cards.find((item) => item.dataset.stage === stage);

  points.forEach((item) => item.classList.toggle("is-active", item === point));
  cards.forEach((item) => item.classList.toggle("is-active", item === card));

  if (!point) return;
  tooltip.innerHTML = `
    <strong>${point.dataset.stage} · ${point.dataset.value} TFLOPS</strong>
    <span>${point.dataset.note}</span>
  `;
}

points.forEach((point) => {
  point.setAttribute("tabindex", "0");
  point.setAttribute(
    "aria-label",
    `${point.dataset.stage}，${point.dataset.value} TFLOPS，${point.dataset.note}`,
  );

  point.addEventListener("mouseenter", () => selectStage(point.dataset.stage));
  point.addEventListener("focus", () => selectStage(point.dataset.stage));
  point.addEventListener("click", () => selectStage(point.dataset.stage));
  point.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectStage(point.dataset.stage);
    }
  });
});

cards.forEach((card) => {
  card.addEventListener("mouseenter", () => selectStage(card.dataset.stage));
  card.addEventListener("focus", () => selectStage(card.dataset.stage));
  card.addEventListener("click", () => selectStage(card.dataset.stage));
});
