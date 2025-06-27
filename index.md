---
layout: default
title: Welcome to My Blog
---

# Welcome to my blog!

This is the homepage of my GitHub Pages site. You can find my posts in the [posts section](/posts).

## Latest Post

{% for post in site.posts limit:1 %}
- [{{ post.title }}]({{ post.url | relative_url }})
{% endfor %}
