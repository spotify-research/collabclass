# collabclass

Reproducibility package for the paper:

> Lucas Maystre, Nagarjuna Kumarappan, Judith Bütepage, Mounia Lalmas.
> _[Collaborative Classification from Noisy
> Labels](http://proceedings.mlr.press/v130/maystre21a.html)_, AISTATS 2021.

This repository contains

- a reference implementation of the algorithms presented in the paper, and
- Jupyter notebooks enabling the reproduction of some of the experiments.


## Getting started

Our codebase was tested with Python 3.8. The following libraries are required:

- `numpy` (tested with version 1.19.2)
- `scipy` (tested with version 1.6.2)
- `matplotlib` (tested with version 3.3.4)
- `numba` (tested with version 0.53.1)
- `notebook` (tested with version 6.3.0)

To get started, follow these steps:

- Clone the repo locally with: `git clone
  https://github.com/spotify-research/collabclass.git`
- Move to the repository: `cd collabclass`
- Install the dependencies: `pip install -r requirements.txt`
- Install the package: `pip install -e lib/`
- Move to the notebook folder: `cd notebooks`
- Start a notebook server: `jupyter notebok`


## Support

Create a [new issue](https://github.com/spotify-research/collabclass/issues/new)


## Contributing

We feel that a welcoming community is important and we ask that you follow
Spotify's [Open Source Code of
Conduct](https://github.com/spotify/code-of-conduct/blob/master/code-of-conduct.md)
in all interactions with the community.


## Author

[Lucas Maystre](mailto:lucasm@spotify.com)

A full list of [contributors](https://github.com/spotify-research/cosernn/graphs/contributors?type=a) can
be found on GitHub.

Follow [@SpotifyResearch](https://twitter.com/SpotifyResearch) on Twitter for
updates.


## License

Copyright 2021 Spotify, Inc.

Licensed under the Apache License, Version 2.0:
https://www.apache.org/licenses/LICENSE-2.0


## Security Issues?

Please report sensitive security issues via Spotify's bug-bounty program
(https://hackerone.com/spotify) rather than GitHub.
