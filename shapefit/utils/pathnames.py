import os


class AnyPathnames:
    SHAPENET_OBJ_IN_SCANNET_SCANS_PATH = 'Shapenet_objects_in_Scannet_scans(Scan2CAD).json'

    SHAPES_IN_SCAN2CAD_PARTS = 'PartNet_dir_of_ShapeNet_objects.json'

    PART_ID_TO_PARTS_DESCRIPTION = 'part_id_to_parts_description.csv'

    APPEARANCES_ALL = 'cad_appearances.json'
    APPEARANCES_2K = 'partnet_validation_2k.json'
    APPEARANCES_50 = 'partnet_validation_50.json'


class ContainerPathnames(AnyPathnames):
    # assumes that we run in container
    DICTIONARIES_PATH = '/data/dictionaries'

    SHAPENET_OBJ_IN_SCANNET_SCANS_PATH = os.path.join(
        DICTIONARIES_PATH, AnyPathnames.SHAPENET_OBJ_IN_SCANNET_SCANS_PATH)

    SHAPES_IN_SCAN2CAD_PARTS = os.path.join(
        DICTIONARIES_PATH, AnyPathnames.SHAPES_IN_SCAN2CAD_PARTS)

    PART_ID_TO_PARTS_DESCRIPTION = os.path.join(
        DICTIONARIES_PATH, AnyPathnames.PART_ID_TO_PARTS_DESCRIPTION)

    APPEARANCES_ALL = os.path.join(
        DICTIONARIES_PATH, AnyPathnames.APPEARANCES_ALL)
    APPEARANCES_2K = os.path.join(
        DICTIONARIES_PATH, AnyPathnames.APPEARANCES_2K)
    APPEARANCES_50 = os.path.join(
        DICTIONARIES_PATH, AnyPathnames.APPEARANCES_50)



class LocalPathnames(AnyPathnames):
    # assumes that we run in container
    DICTIONARIES_PATH = os.path.realpath('/home/ishvlad/workspace/Scan2CAD/Assets/full/dictionaries')

    SHAPENET_OBJ_IN_SCANNET_SCANS_PATH = os.path.join(
        DICTIONARIES_PATH, AnyPathnames.SHAPENET_OBJ_IN_SCANNET_SCANS_PATH)

    SHAPES_IN_SCAN2CAD_PARTS = os.path.join(
        DICTIONARIES_PATH, AnyPathnames.SHAPES_IN_SCAN2CAD_PARTS)

    PART_ID_TO_PARTS_DESCRIPTION = os.path.join(
        DICTIONARIES_PATH, AnyPathnames.PART_ID_TO_PARTS_DESCRIPTION)

    APPEARANCES_ALL = os.path.join(
        DICTIONARIES_PATH, AnyPathnames.APPEARANCES_ALL)
    APPEARANCES_2K = os.path.join(
        DICTIONARIES_PATH, AnyPathnames.APPEARANCES_2K)
    APPEARANCES_50 = os.path.join(
        DICTIONARIES_PATH, AnyPathnames.APPEARANCES_50)
