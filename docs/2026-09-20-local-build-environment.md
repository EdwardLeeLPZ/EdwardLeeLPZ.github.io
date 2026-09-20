# 本地构建环境：诊断与方案

日期：2026-09-20
状态：已实测通过

## 1. 结论

本机系统 Ruby 无法构建本站点，且无法通过升级 bundler 或修改 `Gemfile.lock` 绕过。本地构建改为在容器内使用与部署工作流完全相同的 Ruby 版本，入口是 `tools/jekyll-docker.sh`。系统 Ruby 与 `Gemfile.lock` 均未改动。

## 2. 故障现象

`bundle install` 报：

```
Your bundle is locked to nokogiri (1.18.8-x86_64-linux-musl) from rubygems
repository http://rubygems.org/ or installed locally, but that version can no
longer be found in that source.
```

## 3. 根因

三条事实叠加，每条都单独致命：

| 事实                                                                                                                                       | 证据                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| 系统 Ruby 为 3.0.2p107，bundler 为随 Ruby 分发的 default gem 2.2.22                                                                        | `ruby --version`、`gem list bundler`                             |
| `Gemfile.lock` 由 bundler 2.5.23 生成，部署工作流使用 Ruby 3.3.5                                                                           | `Gemfile.lock` 的 `BUNDLED WITH`、`deploy.yml` 的 `ruby-version` |
| bundler 2.2.22 不区分 `linux-gnu` 与 `linux-musl` 平台，把本机的 `x86_64-linux` 错配到 lockfile 中的 `nokogiri (1.18.8-x86_64-linux-musl)` | 上述报错信息                                                     |

即使修正平台匹配也无法解决：`nokogiri` 1.18.x 要求 Ruby >= 3.1，而本机是 3.0.2。因此**必须更换 Ruby 运行时**，这是根因而非表象。

另需注意，本机的 bundler 可执行文件名为 `bundle3.0` / `bundler3.0`，不存在 `bundle`。

## 4. 方案选择

| 方案                                    | 取舍                                                                                             |
| --------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 升级系统 bundler                        | 无效。解决不了 nokogiri 对 Ruby >= 3.1 的要求。                                                  |
| 修改 `Gemfile.lock` 降级依赖            | 否决。该文件被 git 跟踪，且 `deploy.yml` 的 `paths` 过滤器监听它，改动会影响线上部署。           |
| 用 rbenv / ruby-build 装原生 Ruby 3.3.5 | 否决。需要 sudo 安装一批编译依赖并编译约 10 分钟，且污染主机环境。                               |
| **容器内运行 `ruby:3.3.5`**             | **采用。** 与部署工作流版本完全一致，不动主机，Linux 下 bind mount 为原生 I/O，构建耗时约 2 秒。 |

## 5. 实现

`tools/jekyll-docker.sh`，子命令 `install` / `check` / `build` / `serve` / `shell` / `run`。

关键设计点：

- 镜像标签 `ruby:3.3.5` 写在脚本顶部单一变量中，注释要求与 `deploy.yml` 的 `ruby-version` 保持同步。
- 以 `--user $(id -u):$(id -g)` 运行，保证 `_site/`、`vendor/` 的属主是当前用户而非 root。
- gem 安装到 `vendor/bundle`。该路径已在 `.gitignore` 中，也已在 `_config.yml` 的 `exclude` 中，因此既不入库也不会被发布，且跨次运行复用，只下载一次。
- `build` 与 `serve` 先执行 `bundle check`，失败才 `bundle install`，两步共用同一个容器，常见情形下只有一次容器启动开销。
- `--config` 列表中的 `_config.local.yml` 仅在文件存在时追加。实测 Jekyll 遇到 `--config` 列表里缺失的文件会 `Fatal` 退出（exit 1），而非仅告警。
- `tools` 已加入 `_config.yml` 的 `exclude`。`_scripts/` 不可用于存放此类脚本，因为它在 `_config.yml` 的 `include: ["_pages", "_scripts"]` 中，会被发布到公网。
- `serve` 启动前预检端口占用。本机存在一个属主非当前用户的进程长期监听 `0.0.0.0:4000`（`ss -ltnp` 查不到 PID），默认端口因此不可用；预检会直接提示改用其它端口，而不是让 Docker 在准备完容器后才报 `address already in use`。端口通过参数传入，例如 `tools/jekyll-docker.sh serve 4321`。

## 6. 实测结果

```
$ npx prettier . --check
All matched files use Prettier code style!

$ tools/jekyll-docker.sh check      # exit 0
The Gemfile's dependencies are satisfied

$ tools/jekyll-docker.sh build      # exit 0
Configuration file: /srv/_config.yml
            Source: /srv
       Destination: /srv/_site
                    done in 2.363 seconds.

$ tools/jekyll-docker.sh serve 4321
    Server address: http://0.0.0.0:4321
 LiveReload address: http://0.0.0.0:35729
  Server running... press ctrl-c to stop.
```

产物核对：`_site/` 属主为当前用户；`_site/docs`、`_site/tools`、`_site/vendor` 均不存在；仓库内无 root 属主文件。

`serve` 起服务后约 8 秒可访问，实测 `GET /`、`GET /publications/`、`GET /cv/` 均返回 200，`GET /tools/jekyll-docker.sh` 返回 404，确认脚本未被发布。

构建日志中的 Sass deprecation 警告来自 vendored 的 font-awesome 与 tabler-icons，属既有问题，与本次改动无关。

## 7. 顺带发现并一并修复的既有问题

诊断过程中查出三处既有缺陷，均已修复：

1. **`README.md` 记录的验证命令在干净检出上必然失败。** 原命令为 `bundle exec jekyll build --config _config.yml,_config.local.yml --disable-disk-cache`，但 `_config.local.yml` 被 `.gitignore` 忽略、默认不存在。实测 Jekyll 遇到 `--config` 列表中缺失的文件会 `Fatal` 退出（exit 1），并非仅告警。已改为 `tools/jekyll-docker.sh` 的两条命令，并在 README 中写明 `_config.local.yml` 只有在文件存在时才可传入。
2. **`Gemfile.lock` 同时被 git 跟踪、又列在 `.gitignore` 中。** `.gitignore` 对已跟踪文件不生效，所以此前行为正确，但配置自相矛盾；且 `deploy.yml` 的 `paths` 过滤器专门监听 `Gemfile.lock`，说明它本就应当被跟踪。已删除 `.gitignore` 中的该行。
3. **`deploy.yml` 每次部署都执行 `apt-get update && apt-get install -y imagemagick`，但 `_config.yml` 中 `imagemagick.enabled: false`。** 该步骤不产生任何效果，只增加每次部署的耗时。已删除，同时在 `_config.yml` 的 `imagemagick` 配置旁注明：日后若启用该开关，需同步恢复 `deploy.yml` 中的安装步骤。
