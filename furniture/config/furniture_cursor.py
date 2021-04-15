def add_argument(parser):
    parser.set_defaults(alignment_pos_dist=0.1)
    parser.set_defaults(alignment_rot_dist_up=0.9)
    parser.set_defaults(alignment_rot_dist_forward=0.9)
    parser.set_defaults(alignment_project_dist=0.)
