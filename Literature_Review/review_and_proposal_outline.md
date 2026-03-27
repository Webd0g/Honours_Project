# Review and Proposal Outline

---

## Introduction
- Define transients and their astrophysical importance, with SNe Ia as the primary focus given their role as standardisable candles for cosmological distance measurement.
- Establish the cosmological motivation: why precise SN Ia standardisation matters for measurements of H₀ and the dark energy equation of state. (Need reference: Phillips 1993 or Perlmutter/Riess foundational papers.)
- Introduce the current limits on SN Ia standardisation — residual scatter in the Hubble diagram driven by environmental and progenitor diversity — as the scientific problem this project addresses.
- How Rubin/LSST will transform transient discovery volume, and why this scale requires automated broker tools like Fink to do real-time science with the alert stream.

---

## SN Ia Physics and Progenitor Channels

### Standard Candle Model and Standardisation
- Explain the Phillips relation and how light-curve width and colour corrections (SALT2/MLCS) are used to standardise SN Ia luminosities.
- Introduce the residual systematics: the host galaxy mass step (Rigault et al. 2020; Wiseman et al. 2023), sSFR dependence, and colour/dust effects (Kelsey et al. 2022; Ramaiya et al. 2025).
- Establish that these correlations with host properties are not fully corrected by standard light-curve fitting, motivating the need to understand their physical origin.

### Progenitor Channels
- Introduce the single degenerate (SD) and double degenerate (DD) progenitor channels and their key observational differences. (Need reference: Maoz, Mannucci & Nelemans 2014 review.)
- Discuss delay time distributions and how they connect host galaxy stellar populations (star formation rate, stellar mass, age) to the mix of progenitor types — this is the physical bridge between host environment and explosion physics.
- Note that different progenitor channels predict different $^{56}$Ni distributions, ejecta structure, and interaction signatures, all of which manifest in the early light curve.

### Early-Time Light Curve Behaviour and the Early Flux Excess
- Explain why the first few days post-explosion are the critical diagnostic window for progenitor channel identification.
- Describe the physical mechanisms that can produce an early flux excess (EEx): (1) extended/surface $^{56}$Ni distribution, (2) ejecta-companion interaction (Kasen 2010), (3) CSM interaction, (4) He-shell detonation (relevant to 02es-like events, Srivastav et al. 2023).
- Explain specifically how different $^{56}$Ni distributions change the shape of the early light curve — this is the key physical concept for the scientific thoroughness criterion.
- Summarise Deckers et al. (2022) findings: EEx rate of ~18±11% at z<0.07 using ZTF, higher frequency in 91T/99aa-like events (44±13%), and the constraints placed on progenitor models. This is the primary methodological reference for the project.

---

## Host Galaxy Environment and SN Ia Diversity

### Global Host Properties and the Mass Step
- Summarise what is known about where SNe Ia occur: radial distributions within host galaxies (Pritchet et al. 2024; Hakobyan et al. 2016), and how SNe Ia broadly trace host stellar light (Anderson et al. 2015).
- Discuss the host galaxy mass step — SNe Ia in more massive, passive hosts appear systematically brighter post-standardisation — and the evidence that stellar population age is the underlying driver (Wiseman et al. 2023; Rigault et al. 2020).
- Cover colour and dust effects: Kelsey et al. (2022) on how the mass step varies with SN colour, and Ramaiya et al. (2025) on the SFR–M★ plane as a framework for grouping host environments.

### Local Environment
- Distinguish between global host properties (total stellar mass) and local environment (sSFR within ~1 kpc of the SN site) — Rigault et al. (2020) show local sSFR is a stronger predictor of standardised brightness than global mass.
- Martin et al. (2024): [O II] EW as a local sSFR indicator and its effectiveness at removing the mass step.
- de Sá-Freitas et al. (2021): star formation quenching and morphological transformation — provides context for interpreting passive vs star-forming host subsamples.

### Connecting Environment to Early Excess
- **This is the key synthesis point for the scientific concepts criterion.** 
- 91T/99aa-like events — which Deckers et al. find have the highest EEx rates — preferentially occur in specific host environments. If EEx events are non-uniformly distributed across the host population, they may be systematically biasing the subsamples used in standardisation.
- Identify the gap in the literature: EEx rates as a function of host galaxy properties (sSFR, stellar mass, colour, morphology) have not been systematically characterised. This is what the project will address.

---

## Surveys, Brokers and Data Infrastructure

- Brief overview of Rubin/LSST: survey parameters, expected alert volume (~10 million alerts/night), and why this necessitates broker-level automation.
- Fink broker: architecture, real-time classification pipeline, and how it connects to the Rubin alert stream (Möller et al. 2021). Keep this section lean — it is methodology context, not core science.
- Fink's SN Ia classification modules: active learning approach (Leoni et al. 2022) and LSST benchmark performance (Fraga et al. 2024).
- ZTF as the precursor survey: Deckers et al. (2022) used ZTF data — establishing continuity between ZTF methodology and the LSST era is important for the methodology section.

---

## Project Aims
- State the primary aim: to detect and characterise early flux excess SNe Ia in the Rubin/LSST alert stream, processed via the Fink broker, and to investigate how their occurrence rates correlate with host galaxy properties.
- Secondary aim: to use the EEx sample to constrain viable progenitor channels, following the approach of Deckers et al. (2022), and assess whether the EEx population represents a systematic bias in current standardisation frameworks.
- **Include an explicit caveat**: the EEx detection component is contingent on Fink receiving and processing suitable Rubin alerts during the project period. Given LSST's expected detection volume this is highly probable, but the methodology is designed to be robust to this — if the live alert sample is limited, archival ZTF data can supplement the analysis.

---

## Planned Methodology
- **Sample construction**: Use Fink's SN Ia classification modules (active learning classifier, Leoni et al. 2022) to identify SN Ia candidates from the Rubin alert stream in real time. Supplement with archival ZTF data if needed.
- **EEx identification**: Apply the Deckers et al. (2022) methodology — fit the early light curves (pre-maximum, first ~5 days) using the `turtls` model grid to identify flux excesses in gr-band data, distinguishing blue and red excess events.
- **Spectroscopic and multi-wavelength classification**: Use available spectroscopic follow-up and multi-band photometry (accessible via Fink's cross-match with external catalogues) to confirm SN type and identify 91T/99aa-like vs normal events.
- **Host galaxy association**: Cross-match SN positions with host galaxy catalogues to determine host properties: stellar mass, sSFR, U-R colour, and morphology. Compute fractional host-light offsets (following Pritchet et al. 2024; Hakobyan et al. 2016) to characterise the SN's location within its host.
- **Environment–EEx correlation analysis**: Compare EEx rates across host property bins (e.g. passive vs star-forming, high vs low stellar mass, early vs late morphological type). Use this to assess whether EEx events preferentially occur in specific environments and what that implies for progenitor channel fractions.
- **Standardisation implications**: Assess whether the identified EEx subsample shows a statistically distinct mass step or sSFR dependence compared to the broader SN Ia sample, connecting back to the standardisation motivation.

---

## References
