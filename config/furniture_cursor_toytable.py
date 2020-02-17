from util import str2bool, str2intlist
from env.models import furniture_names, furniture_ids, background_names

def add_argument(parser):
    parser.add_argument('--site_dist_rew', type=float, default=1, help='rew for dist between sites')
    parser.add_argument('--site_up_rew', type=float, default=1, help='rew for angular dist between site up vecs')
    parser.add_argument('--connect_rew', type=float, default=10, help='rew for connecting')
    parser.add_argument('--success_rew', type=float, default=100, help='rew for successful connect')
    parser.add_argument('--pick_rew', type=float, default=1, help='rew for successful pick')
