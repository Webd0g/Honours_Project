Webd0g<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />

<h3 align="center">Honours Project</h3>

  <p align="center">
    This is the code that was used for the database filtering and model fitting throughout my Honours Project at Swinburne University. The aims of the Honours project were to first find a sample of extragalatic SN-like objects, then narrow that sample down to objects that are consistent with properties of real SNe Ia using the SALT3 SED model. Then using that sample of SNe Ia, use some established empirical methods to find EEx SNe Ia. The last aims was that if a good and reasonable EEx SNe Ia object was found, we could then use the galaxy that it was cross-matched with to find some host properties of the galaxy which could contribute to growing the niche field of EEx SNe Ia that have been cross-matched with some identifiable hosts.  
    <br />
    <a href="https://github.com/Webd0g/Honours_Project"><strong>Explore the docs »</strong></a>
    <br />
    &middot;
    <a href="https://github.com/Webd0g/Honours_Project/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/Webd0g/Honours_Project/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
* [![Polars](https://img.shields.io/badge/Polars-DataFrames-blue)](https://pola.rs/)
* [![Astropy](https://img.shields.io/badge/Astropy-v5.0+-orange)](https://www.astropy.org/)
* [![SNCosmo](https://img.shields.io/badge/SNCosmo-SALT3-red)](https://sncosmo.readthedocs.io/)
* [![emcee](https://img.shields.io/badge/emcee-MCMC-green)](https://emcee.readthedocs.io/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
- Python 3.10+
- `conda` or `venv` package manager

### Installation

1. Clone the repo
```bash
   git clone [https://github.com/Webd0g/Honours_Project.git](https://github.com/Webd0g/Honours_Project.git)
   cd Honours_Project  
```

2. Create and activate a virtual environment
```bash 
    python -m venv .venv
    source .venv/bin/activate
```

3. Install requirements and package in editable mode
```bash 
  pip install -r requirements.txt
  pip install -e
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

`rubin_eex_finder` provides automated modules for filtering alert stream light curves, running empirical (Bazin, SALT3) and power-law MCMC model fits, and extracting early excess (EEx) candidates.

### Running the Early-Excess Power-Law Model Fit

The example below demonstrates how to load processed rest-frame light curves, exclude points inside the early excess window, fit a multi-band power law using `emcee` (100 walkers, 2000 steps), and plot the candidate results:

```python
from pathlib import Path
import polars as pl
from rubin_eex_finder.models.excess_power_law import (
    excluding_excess_window,
    fitting_power_law,
    fit_sample_parameters_plot,
)

# 1. Load rest-frame converted light curve data
data_dir = Path("data/eex_results/simple_power_law")
full_rest_data = pl.read_csv("data/rest_frame_light_curves.csv")

# 2. Exclude observations inside the 3-day early excess window (t_fd to t_fd + 3d)
rest_data_no_eex = excluding_excess_window(
    rest_frame_conversion_df=full_rest_data,
    window_length=3,
)

# 3. Fit the multi-band simple power law via MCMC
fit_results, sampler = fitting_power_law(
    rest_frame_objects=rest_data_no_eex,
    n_steps=2000,
    fit_df_output_path=data_dir / "results/3_day_window/pass.csv",
)

# 4. Generate diagnostic plots (light curve + residual pull panels)
fit_sample_parameters_plot(
    pl_fit_params_df=fit_results,
    rest_frame_no_eex=rest_data_no_eex,
    window_length=3,
    full_rest_data_df=full_rest_data,
    output_path=Path("images/early_excess_plots/3_day_window"),
)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/Webd0g/Honours_Project/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Webd0g/Honours_Project" alt="contrib.rocks image" />
</a>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

```markdown
Riley Webster - webd0g@duck.com

Project Link: [https://github.com/Webd0g/Honours_Project](https://github.com/Webd0g/Honours_Project)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
```

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

```markdown
* **Dr. Anais Möller** (Swinburne University of Technology) - For invaluable guidance, support and expertise throughout the project.
* **Brayden Leicester & Lukas Steinwender** - For asking interesting questions about any and all results I presented and useful feedback throughout the project.
* **Family & Friends** - For their constant encoruagement, patience and unwavering belief during the project.
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Webd0g/Honours_Project.svg?style=for-the-badge
[contributors-url]: https://github.com/Webd0g/Honours_Project/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Webd0g/Honours_Project.svg?style=for-the-badge
[forks-url]: https://github.com/Webd0g/Honours_Project/network/members
[stars-shield]: https://img.shields.io/github/stars/Webd0g/Honours_Project.svg?style=for-the-badge
[stars-url]: https://github.com/Webd0g/Honours_Project/stargazers
[issues-shield]: https://img.shields.io/github/issues/Webd0g/Honours_Project.svg?style=for-the-badge
[issues-url]: https://github.com/Webd0g/Honours_Project/issues
[license-shield]: https://img.shields.io/github/license/Webd0g/Honours_Project.svg?style=for-the-badge
[license-url]: https://github.com/Webd0g/Honours_Project/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/riley-webster-714468268
[product-screenshot]: images/screenshot.png
<!-- Shields.io badges. You can a comprehensive list with many more badges at: https://github.com/inttter/md-badges -->
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 

