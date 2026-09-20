---
layout: page
permalink: /projects/
title: Projects
description: Engineering demos, system prototypes, and internal tooling.
nav: true
nav_order: 3
---

<div class="project-list">
  {% assign sorted_projects = site.projects | sort: "importance" %}
  {% for project in sorted_projects %}
    {% include project_entry.liquid project=project %}
  {% else %}
    <p class="project-list__empty">Nothing published here yet.</p>
  {% endfor %}
</div>
