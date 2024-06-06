from uhc.data_loaders.mjxml.MujocoXML import MujocoXML

import loguru


def merge_model(hand_model_file, obj_model_file):
    hand_model = MujocoXML(hand_model_file)
    obj_model = MujocoXML(obj_model_file)
    ref_obj_model = MujocoXML(obj_model_file)
    # ref_obj_model2 = MujocoXML(obj_model_file)

    # ref_obj_model.worldbody.getchildren()[0].attrib['name'] += '_ref'
    wb_list = list(ref_obj_model.worldbody)
    wb_list[0].attrib['name'] += '_ref'
    for ele in wb_list[0]:
        if ele.tag == 'geom':
            ele.attrib['material'] = "object_ref"
            ele.attrib['name'] += "_ref"
            # ele.attrib['group'] = "2"

    hand_model.merge(obj_model)
    hand_model.merge(ref_obj_model)

    # get object mesh file
    obj_mesh_fn = None
    for c in obj_model.asset:
        if c.get('name').startswith('V_'):
            obj_mesh_fn = c.get('file')
            break
    if obj_mesh_fn is None:
        loguru.logger.error("Error: No Object Mesh file Found!")
        raise FileNotFoundError

    return hand_model.get_xml(), obj_mesh_fn


def merge_model_from_cfg(cfg):
    return merge_model(cfg.vis_model_file, cfg.data_specs['obj_fn'])
