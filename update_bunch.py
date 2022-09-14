def update_bunch(bunch,intensity,
                bunch_dict,beta,gamma,p0):
    bunch.update(bunch_dict)
    bunch.beta = beta
    bunch.gamma = gamma
    bunch.intensity = intensity
    bunch.p0 = p0