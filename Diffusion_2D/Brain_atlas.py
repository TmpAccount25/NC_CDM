import siibra
from nilearn import plotting

if __name__ == "__main__":
    julich_pmaps = siibra.get_map(
        parcellation="julich 2.9",
        space="mni152",
        maptype="statistical"
    )
