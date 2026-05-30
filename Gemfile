source 'http://rubygems.org'

gem 'jekyll'

# Core plugins that directly affect site building
group :jekyll_plugins do
    gem 'jekyll-archives-v2'
    gem 'jekyll-email-protect'
    gem 'jekyll-feed'
    gem 'jekyll-link-attributes'
    gem 'jekyll-minifier'
    gem 'jekyll-paginate-v2'
    gem 'jekyll-regex-replace'
    gem 'jekyll-scholar'
    gem 'jekyll-sitemap'
    gem 'jekyll-toc'
    gem 'jemoji'

    gem 'classifier-reborn'  # used for content categorization during the build
end

# Gems for development or external data fetching (outside :jekyll_plugins)
group :other_plugins do
    gem 'css_parser'
    gem 'feedjira'
    gem 'httparty'
    gem 'observer'       # used by jekyll-scholar
    gem 'ostruct'
    # gem 'unicode_utils' -- should be already installed by jekyll
    # gem 'webrick' -- should be already installed by jekyll
end

group :development do
    gem 'wdm', '>= 0.1.1', :platforms => [:mingw, :x64_mingw, :mswin]
end