import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import time
import glob
import json
from tqdm import tqdm
from dataset import *
from models.image_nuscenes_multi3dpred import ImagePolicyModelSS
from params import par
from utils.train_utils import one_hot
from nusc_scene_map import nusc_scene_map

class Metrics(torch.nn.Module):
    def __init__(self, rg=25, offset=25, device='cuda', **kwargs):
        super().__init__()
        self._norm = torch.FloatTensor([rg, rg]).to(device)
        self.offset = offset

    def forward(self, wpts_est, wpts_gt):
        wpts_est[:,:,1] = wpts_est[:,:,1] + 1
        wpts_est = wpts_est * self._norm

        return torch.abs(wpts_est-wpts_gt).sum(axis=2).mean(axis=1), torch.sqrt(((wpts_est-wpts_gt)**2).sum(axis=2)), torch.sqrt(((wpts_est-wpts_gt)**2).sum(axis=2))[:, -1]
    
metrics = Metrics()

def _eval(model, test_dl):

    model.eval()
    eval_loss_list = []

    with torch.no_grad():
        for step, (rgb_img, rgb_img_pth, cmd, speed, wpts_gt, _) in enumerate(test_dl):
            rgb_img, cmd, speed, wpts_gt = rgb_img.to('cuda'), one_hot(cmd).to('cuda'), speed.type(torch.float32).to('cuda'), wpts_gt.to('cuda')
            wpts_est, _ = model(rgb_img, cmd, speed)
            
            _, ade_l2, fde_l2 = metrics(wpts_est, wpts_gt)
            _wpts_gt = wpts_gt.cpu().detach().numpy()
            _ade_l2 = ade_l2.cpu().detach().numpy()
            _fde_l2 = fde_l2.cpu().detach().numpy()
            for num, _ in enumerate(_wpts_gt):
                eval_loss_list.append([_wpts_gt[num], _ade_l2[num].mean(), _fde_l2[num]])
                
    return eval_loss_list


def eval(train_scenes, start_ep):
    model = ImagePolicyModelSS(par.backbone, pretrained=par.imagenet_pretrained)
    model = model.to('cuda')
    
    if start_ep == -1:
        metrics_result = {}
    else:
        metrics_result = np.load('./results/{}/metrics_result.npy'.format(par.json_file.split('.')[0]), allow_pickle=True).item()
    sets = ['NUSC', 'KITTI', 'ARGO2']
    degrees = ['left', 'right', 'straight']

    for _scene in train_scenes:
        
        if start_ep == -1:
            pass
        elif int(_scene[-3:]) < start_ep:
            continue
        
        metrics_result[_scene] = {}
        for _set in sets:
            metrics_result[_scene][_set] = {}
            # metrics_result[_scene][_set]['ave_ade'] = []
            # metrics_result[_scene][_set]['ave_fde'] = []
            for _deg in degrees:
                metrics_result[_scene][_set][_deg] = {}
                
        pt_list = sorted(glob.glob(par.model_path + '/{}/*.pt'.format(_scene)))
        if len(pt_list) != par.epochs:
            print('Stop evaluating scene {}.'.format(_scene))
            assert 0
        
        for pt in pt_list:
            print('Restored model from {}:'.format(pt))
            
            checkpoint = torch.load(pt)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            for _set in sets:
                
                _temp = {}
                for _deg in degrees:
                    _temp[_deg] = {}
                    
                if _set == 'NUSC':
                    eval_path = par.data_pth['NUSC']
                    Scenes = nusc_scene_map['singapore-onenorth'] + nusc_scene_map['singapore-queenstown'] + nusc_scene_map['singapore-hollandvillage']
                elif _set == 'KITTI':
                    eval_path = par.data_pth['KITTI']
                    Scenes = [str(i).zfill(2) for i in range(11)]
                elif _set == 'ARGO2':
                    eval_path = par.data_pth['ARGO2'] + '_Test'
                    Scenes = [str(i).zfill(3) for i in range(150)]
                    
                test_df = get_data_info(eval_path, eval_path+'/poses', Scenes)
                test_dataset = ImageDataset(test_df, train=False)
                test_dl = DataLoader(
                    test_dataset, 
                    batch_size=par.batch_size, 
                    shuffle=False, 
                    num_workers=par.n_processors,
                    pin_memory=True,
                )
                
                eval_loss_list = _eval(model, test_dl)
                # _ade_list = []
                # _fde_list = []
                
                for _eval_loss in eval_loss_list:
                    sample_wpts = _eval_loss[0]
                    _ade = _eval_loss[1]
                    _fde = _eval_loss[2]
                    # _ade_list.append(_ade)
                    # _fde_list.append(_fde)
                    future_x, future_y = sample_wpts[-1]
                    
                    speed = np.sqrt(sample_wpts[0][0]**2 + sample_wpts[0][1]**2)*2
                    key = math.floor(speed)
                    if key >= 16:
                        key = 16
                    
                    if future_y == 0:
                        assert key == 0
                        if key not in _temp['straight'].keys():
                            _temp['straight'][key] = {'ade': [], 'fde': []}
                        _temp['straight'][key]['ade'].append(_ade)
                        _temp['straight'][key]['fde'].append(_fde)
                    else:
                        rad = math.atan2(future_y, future_x)
                        if rad >= 0 and rad < (math.pi * 85 / 180):
                            case = 'right'
                        elif rad >= (math.pi * 85 / 180) and rad < (math.pi * 95 / 180):
                            case = 'straight'
                        elif rad >= (math.pi * 95 / 180) and rad <= (math.pi * 180 / 180):
                            case = 'left'
                            
                        if key not in _temp[case].keys():
                            _temp[case][key] = {'ade': [], 'fde': []}
                        _temp[case][key]['ade'].append(_ade)
                        _temp[case][key]['fde'].append(_fde)
                        
                # metrics_result[_scene][_set]['ave_ade'].append(sum(_ade_list)/len(_ade_list))
                # metrics_result[_scene][_set]['ave_fde'].append(sum(_fde_list)/len(_fde_list))
                        
                for _key1 in _temp.keys():
                    for _key2 in _temp[_key1].keys():
                        if _key2 not in metrics_result[_scene][_set][_key1].keys():
                            metrics_result[_scene][_set][_key1][_key2] = {'ade': [], 'fde': []}
                        metrics_result[_scene][_set][_key1][_key2]['ade'].append(sum(_temp[_key1][_key2]['ade']) / len(_temp[_key1][_key2]['ade']))
                        metrics_result[_scene][_set][_key1][_key2]['fde'].append(sum(_temp[_key1][_key2]['fde']) / len(_temp[_key1][_key2]['fde']))
                        
                np.save('./results/{}/metrics_result.npy'.format(par.json_file.split('.')[0]), metrics_result)


if __name__ == '__main__':

    train_scenes = list(par.video_streaming.keys())
    eval(train_scenes, start_ep = 50)