def determine_health_status(ndvi, vari, gli, exg):
    # Example conditions for determining health status
    avg_ndvi = ndvi.mean()
    avg_vari = vari.mean()
    avg_gli = gli.mean()
    avg_exg = exg.mean()

    # Set thresholds for healthy/unhealthy classification
    if avg_ndvi > 0.5 and avg_vari > 0.2 and avg_gli > 0.1 and avg_exg > 0.1:
        return "Healthy"
    else:
        return "Unhealthy"
