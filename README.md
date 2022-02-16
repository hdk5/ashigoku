# This tools has been RiiR'd

Go see [ugokuna](https://github.com/hdk5/ugokuna)

# `ashigoku`

Shit I use to download ugoira illustrations from pixiv as gifs.

It is an awful, horrible shit. I don't like it. I'll RIIR it someday.

Usage:

```sh
ashigoku --help
```

```
Usage: ashigoku [OPTIONS] OUT_DIR

Arguments:
  OUT_DIR  [required]

Options:
  -f, --format [gif|webm]         [default: OutFormat.GIF]
  -a, --artist-id INTEGER         [default: (dynamic)]
  -i, --illust-id INTEGER         [default: (dynamic)]
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.
  --help                          Show this message and exit.
```

```sh
ashigoku -a 16274829 ./out/
```

```sh
ashigoku -i 71720656 -i 71167153 ./out/
```
