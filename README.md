# Deformetrica


Website: [www.deformetrica.org](http://www.deformetrica.org/)

**Deformetrica** is a software for the statistical analysis of 2D and 3D shape data. It essentially computes deformations of the 2D or 3D ambient space, which, in turn, warp any object embedded in this space, whether this object is a curve, a surface, a structured or unstructured set of points, an image, or any combination of them.

_Deformetrica_ comes with three main applications:
- **registration** : estimates the best possible deformation between two sets of objects;
- **atlas construction** : estimates an average object configuration from a collection of object sets, and the deformations from this average to each sample in the collection;
- **geodesic regression** : estimates an object time-series constrained to match as closely as possible a set of observations indexed by time.

_Deformetrica_ has very little requirements about the data it can deal with. In particular, it does __not__ require point correspondence between objects!

## Install

>>>
Warning: Conda packages to install Deformetrica are deprecated from version 4.3.0
To install recent versions of Deformetrica use `pip` (see below).
>>>

- **Requirements**: [Anaconda 3](https://www.anaconda.com/download), Linux or Mac OS X distributions
- **Best practice**: `conda create -n deformetrica python=3.8 numpy && source activate deformetrica`
- **Pip install**: `pip install deformetrica`
- **Run** an [example](https://gitlab.icm-institute.org/aramislab/deformetrica/builds/artifacts/v4.0.0/browse?job=package_and_deploy%3Aexamples): 
    - `deformetrica estimate model.xml data_set.xml --p optimization_parameters.xml`
    - `deformetrica compute model.xml --p optimization_parameters.xml`
- **Try the GUI** (alpha version): `deformetrica gui`
- **Documentation**: [wiki](https://gitlab.icm-institute.org/aramislab/deformetrica/wikis/home)

## Community

- **Need help?** Ask the [Deformetrica Google group](https://groups.google.com/forum/#!forum/deformetrica).
- Spotted an **issue**? Have a **feature request**? Let us know in the [dedicated Gitlab section](https://gitlab.icm-institute.org/aramislab/deformetrica/issues).

## References

Deformetrica relies on a control-points-based instance of the Large Deformation Diffeomorphic Metric Mapping framework, introduced in [\[Durrleman et al. 2014\]](https://linkinghub.elsevier.com/retrieve/pii/S1053811914005205). Are fully described in this article the **shooting**, **registration**, and **deterministic atlas** applications. Equipped with those fundamental building blocks, additional applications have been successively developed:
- the bayesian atlas, described in [\[Gori et al. 2017\]](https://hal.archives-ouvertes.fr/hal-01359423/);
- the geodesic regression, described in [\[Fishbaugh et al. 2017\]](https://www.medicalimageanalysisjournal.com/article/S1361-8415(17)30044-0/fulltext);
- the parallel transport, described in [\[Louis et al. 2018\]](https://www.researchgate.net/publication/319136479_Parallel_transport_in_shape_analysis_a_scalable_numerical_scheme);
- the longitudinal atlas, described in [\[Bône et al. 2018a\]](https://www.researchgate.net/publication/324037371_Learning_distributions_of_shape_trajectories_from_longitudinal_datasets_a_hierarchical_model_on_a_manifold_of_diffeomorphisms) and [\[Bône et al. 2020\]](https://www.researchgate.net/publication/342642363_Learning_the_spatiotemporal_variability_in_longitudinal_shape_data_sets).

[\[Bône et al. 2018b\]](https://www.researchgate.net/publication/327652245_Deformetrica_4_an_open-source_software_for_statistical_shape_analysis) provides a concise reference summarizing those functionalities, with unified notations.

# Archived repositories

- Deformetrica 3: [deformetrica-legacy2](https://gitlab.icm-institute.org/aramislab/deformetrica-legacy2)
- Deformetrica 2.1: [deformetrica-legacy](https://gitlab.icm-institute.org/aramislab/deformetrica-legacy)
