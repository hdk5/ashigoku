# `ashigoku`

Shit I use to download ugoira illustrations from pixiv as gifs.

Usage:

```sh
ashigoku --help
```

```
Usage: ashigoku [OPTIONS] OUT_DIR

Options:
  -a, --artist-id INTEGER
  -i, --illust-id INTEGER
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for
                                  the specified shell.

  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for
                                  the specified shell,
                                  to copy it or
                                  customize the
                                  installation.

  --help                          Show this message and
                                  exit.
```

```sh
ashigoku -a 16274829 ./out/
```

```sh
ashigoku -i 71720656 -i 71167153 ./out/
```
