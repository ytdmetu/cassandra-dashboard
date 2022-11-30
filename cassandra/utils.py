from pathlib import Path


def get_asset_filepath(filename):
    return Path(__file__).parent / "assets" / filename
