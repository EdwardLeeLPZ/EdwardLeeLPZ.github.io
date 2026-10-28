// Fills the star and fork counts on the code repository cards (see
// _includes/repository_list.liquid). One unauthenticated GitHub API request per
// card; a card whose request fails simply keeps its counts hidden.
(function () {
  function format(n) {
    return n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1).replace(/\.0$/, "") + "k" : String(n);
  }

  document.querySelectorAll(".repo-card[data-repo]").forEach(function (card) {
    fetch("https://api.github.com/repos/" + card.dataset.repo, { headers: { Accept: "application/vnd.github+json" } })
      .then(function (response) {
        if (!response.ok) throw new Error(response.status);
        return response.json();
      })
      .then(function (data) {
        card.querySelectorAll("[data-repo-stat]").forEach(function (slot) {
          var value = data[slot.dataset.repoStat];
          if (typeof value !== "number") return;
          slot.querySelector(".repo-card__count").textContent = format(value);
          slot.hidden = false;
        });
      })
      .catch(function () {});
  });
})();
