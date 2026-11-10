# Mock X-ray Observations

A complete mock X-ray analysis of one cluster: its gas is read from the snapshot through Octavius, moxha generates photons from the hot gas and observes them with Chandra ACIS-I (with backgrounds and a blank sky), deprojects the observation into annuli, and fits each annulus's spectrum. The result is a radial profile of temperature, abundance, electron density and X-ray luminosity, each with errors. See [Mock X-ray Observations](../guide/xray.md) for the details.

```{literalinclude} ../../examples/xray_cluster.py
:language: python
```
