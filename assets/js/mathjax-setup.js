window.MathJax = {
  tex: {
    tags: "ams",
    // Upstream al-folio also lists ["$", "$"] here. It is dropped on purpose:
    // kramdown already turns this site's $$...$$ source into \(...\), so the
    // single-dollar delimiter renders no real math, while any prose carrying
    // two dollar amounts gets swallowed into one unwrapped "formula".
    inlineMath: [["\\(", "\\)"]],
  },
  options: {
    renderActions: {
      addCss: [
        200,
        function (doc) {
          const style = document.createElement("style");
          style.innerHTML = `
          .mjx-container {
            color: inherit;
          }
        `;
          document.head.appendChild(style);
        },
        "",
      ],
    },
  },
};
