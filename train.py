import datetime, torch, os, time, sys
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn

from reid import transform_train,transform_auxiliary,transform_test
from reid import LocalSimilarityLoss, LabelSmoothingLoss, GlobalSimilarityLossUnlabeled
from reid import compute_duke_memory_bank
from reid import compute_mutual_knn
from reid import DukeDataProvider, SearchDuke
from reid import MyDataset, DisNet, MyNet
from reid import myprint

MY_GPU = [2]
device = MY_GPU[0]
NEED_MEMORY = 10000
WAIT_LEVEL = 1
SAVE_FILE = '2GanImage'

start_global_at = 5
num_gan_image = 2

EPOCH = 60
pre_epoch = 0
BATCH_SIZE = 32
LR = 0.00035
STEP_SIZE = 20
GAMMA = 0.1
TEST_BATCH_SIZE = 10
Duke_BATCH_SIZE = 16
ALoss_weight = 0.05
LSLoss_weight = 1

GSLoss_weight = 0.1
GSLoss_temperature = 0.05
THRESHOLD = 0.4

# WEIGHT = 'model/epoch5-0.522-0.283.pt'
WEIGHT = 'snapshot/2020-02-08/2GanImage-cluster_0.4-OnlyCluster-005.pt'
FINETUNE = 0
STRICT = 1

BEGIN_TIME = datetime.datetime.now()
today = str(BEGIN_TIME).split()[0]
if not os.path.exists('log/'): os.mkdir('log/')
if not os.path.exists('snapshot/'): os.mkdir('snapshot/')
if not os.path.exists('log/'+today): os.mkdir('log/' + today)
if not os.path.exists('snapshot/'+today): os.mkdir('snapshot/'+today)
LOG_TXT = 'log/' + today + '/' + SAVE_FILE + '.log'
if os.path.exists(LOG_TXT):
    print( 'there already is '+ LOG_TXT +
        '. Press \'y\' or 1 to remove the file and continue. Press other keys to exit.')
    temp_input = input('y/n or 1/0')
    if temp_input == 'y' or temp_input == 1 or temp_input == '1':
        os.remove(LOG_TXT)
    else:
        os._exit(0)

SNAPSHOT = 'snapshot/' + today + '/' + SAVE_FILE + '-'
fid_log = open(LOG_TXT, 'w')
myprint(str(BEGIN_TIME), fid_log)
myprint('Log: ' + LOG_TXT, fid_log)
myprint('Snapshot: ' + SNAPSHOT, fid_log)
myprint("Using GPU: " + str(MY_GPU), fid_log)

free_memory = os.popen('nvidia-smi --query-gpu=memory.free --format=csv,noheader').readlines()[device].split(' ')[0]
free_memory = float(free_memory)
NotOK = 0
if free_memory < NEED_MEMORY:
    NotOK = WAIT_LEVEL
    print('waiting for gpu', device)
while NotOK:
    print(WAIT_LEVEL, end =' ')
    sys.stdout.flush()
    time.sleep(300)
    free_memory = os.popen('nvidia-smi --query-gpu=memory.free --format=csv,noheader').readlines()[device].split(' ')[0]
    free_memory = float(free_memory)
    if free_memory > NEED_MEMORY:
        NotOK -= 1

trainset = MyDataset(root = '/home/xbliu/disk/market1501/', txt = '/home/xbliu/disk/market1501/train.txt', transform = transform_train)
MarketLoader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
DukeLoader = DukeDataProvider(Duke_BATCH_SIZE, num_gan_image = num_gan_image)

net = MyNet()
if FINETUNE:
    net.load_state_dict(torch.load(WEIGHT), strict = STRICT)
net = net.to(device)
net = torch.nn.DataParallel(net, device_ids=MY_GPU, output_device=device)

netD = DisNet().to(device)
netD = torch.nn.DataParallel(netD, device_ids=MY_GPU, output_device=device)

CELoss = LabelSmoothingLoss()
LSLoss = LocalSimilarityLoss()
ALoss = nn.MSELoss()
# GSLossL = GlobalSimilarityLossLabeled()
GSLossU = GlobalSimilarityLossUnlabeled()

optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=2.5e-4)
optimizerD = optim.Adam(netD.parameters(), lr = LR, weight_decay = 2.5e-4)

current_num_iter, max_duke_r1, max_epoch = 0., 0., 0.

BEGIN_TIME = datetime.datetime.now()
for epoch in range(pre_epoch, EPOCH):
    if epoch >= start_global_at:
        if (epoch == start_global_at) or (epoch == pre_epoch) or (epoch%10 == 0):
            duke_memory_bank, _ = compute_duke_memory_bank(net, device = device, feature_index=1)
            duke_memory_bank.requires_grad = False

    if epoch >= start_global_at:
        with torch.no_grad():
            Mutual_KNN = compute_mutual_knn(duke_memory_bank.to('cpu').numpy(), threshold = THRESHOLD )

    this_lr = LR * (GAMMA ** ((epoch+1)//STEP_SIZE))
    myprint('\nEpoch: ' + str(epoch+1) + ', Learning Rate: '+ str(this_lr), fid_log)
    for param_group in optimizer.param_groups: param_group['lr'] = this_lr

    net.train()
    sum_ce_loss, sum_ls_loss, sum_gs_loss, sum_a_loss, sum_d_loss, total, correct = 0., 0., 0., 0., 0., 0., 0.
    
    for sub_iter, market_data in enumerate(MarketLoader, 0):
        current_num_iter += 1

        market_images, market_labels, market_index = market_data
        market_images, market_labels = market_images.to(device), market_labels.to(device)

        DukeImage1, DukeImage2, DukeImage3, DukeIndex = DukeLoader.next()
        DukeImage1, DukeImage2, DukeImage3, DukeIndex = DukeImage1.to(device), DukeImage2.to(device), DukeImage3.to(device), DukeIndex.to(device)

        # if market_images.shape[0]<BATCH_SIZE: 
        #     continue
        if DukeIndex.shape[0] < Duke_BATCH_SIZE:
            continue

        market_mask = market_labels.expand(market_labels.shape[0], market_labels.shape[0]).eq(market_labels.expand(market_labels.shape[0], market_labels.shape[0]).t())
        market_mask.requires_grad = False
        
        DukeIndex = torch.cat((DukeIndex, DukeIndex), 0)
        duke_mask = DukeIndex.expand(DukeIndex.shape[0], DukeIndex.shape[0]).eq(DukeIndex.expand(DukeIndex.shape[0], DukeIndex.shape[0]).t())
        duke_mask.requires_grad = False
        DukeIndex = torch.cat((DukeIndex, DukeIndex[:Duke_BATCH_SIZE]), 0)

        _, market_feature_BN, market_cls = net(market_images)
        # _, market_feature_BN, market_cls = net(market_images[0:4])
        # for n in range(1,8):
        #     _, temp_feature_BN, temp_cls = net(market_images[n*4:n*4+4])
        #     market_feature_BN = torch.cat((market_feature_BN, temp_feature_BN), 0)
        #     market_cls = torch.cat((market_cls, temp_cls), 0)

        _, duke_feature_BN, _ = net(torch.cat((DukeImage1, DukeImage2, DukeImage3),0))

		#label smooth loss
        ce_loss = CELoss(market_cls, market_labels, device=device)
        sum_ce_loss += ce_loss.item()
        _, predicted = torch.max(market_cls.data, 1)
        total += market_labels.size(0)
        correct += predicted.eq(market_labels.data).cpu().sum()

        # normalize the features
        temp_norm = duke_feature_BN.norm(dim=1, keepdim=True)
        temp_norm = temp_norm.expand(duke_feature_BN.shape[0], duke_feature_BN.shape[1])
        duke_feature_BN_normed = duke_feature_BN / temp_norm

        temp_norm = market_feature_BN.norm(dim=1, keepdim=True)
        temp_norm = temp_norm.expand(market_feature_BN.shape[0], market_feature_BN.shape[1])
        market_feature_BN_normed = market_feature_BN / temp_norm

        # update memory bank
        if epoch >= start_global_at:
            duke_memory_bank[DukeIndex[:Duke_BATCH_SIZE]] *= 0.01 * epoch
            duke_memory_bank[DukeIndex[:Duke_BATCH_SIZE]] += duke_feature_BN_normed[:Duke_BATCH_SIZE].detach() * (1 - 0.01 * epoch) /3.
            duke_memory_bank[DukeIndex[:Duke_BATCH_SIZE]] += duke_feature_BN_normed[Duke_BATCH_SIZE:Duke_BATCH_SIZE*2].detach() * (1 - 0.01 * epoch) /3.
            duke_memory_bank[DukeIndex[:Duke_BATCH_SIZE]] += duke_feature_BN_normed[Duke_BATCH_SIZE*2:Duke_BATCH_SIZE*3].detach() * (1 - 0.01 * epoch) /3.

            temp_norm = duke_memory_bank[DukeIndex[:Duke_BATCH_SIZE]].norm(dim=1, keepdim=True)
            temp_norm = temp_norm.expand(Duke_BATCH_SIZE, 2048)
            duke_memory_bank[DukeIndex[:Duke_BATCH_SIZE]] /= temp_norm

        #local similarity loss
        ls_loss = LSLoss(duke_feature_BN_normed[:Duke_BATCH_SIZE*2], duke_mask) * LSLoss_weight
        ls_loss += LSLoss(market_feature_BN_normed, market_mask) * LSLoss_weight
        sum_ls_loss += ls_loss.item()

        if epoch >= start_global_at:
            gs_loss_u = GSLossU(batch_feature = duke_feature_BN_normed[:Duke_BATCH_SIZE*3], batch_index = DukeIndex, memory_feature = duke_memory_bank, GTKNN = Mutual_KNN)
            gs_loss = (gs_loss_u) * GSLoss_weight
            sum_gs_loss += gs_loss.item()
        else:
        	sum_gs_loss = 0.

        D_duke = netD(duke_feature_BN)
        D_market = netD(market_feature_BN)
        a_loss = (ALoss(D_duke, torch.ones_like(D_duke)/2.) + ALoss(D_market, torch.ones_like(D_market))/2.)* ALoss_weight
        sum_a_loss += a_loss.item()

        optimizer.zero_grad()
        if epoch < start_global_at:
            (ce_loss + ls_loss + a_loss).backward()
        else:
            (ce_loss + ls_loss + gs_loss + a_loss).backward()
        optimizer.step()

        # train discriminator
        D_duke = netD(duke_feature_BN.detach())
        D_market = netD(market_feature_BN.detach())
        d_loss = ALoss(D_market, torch.ones_like(D_market)) + ALoss(D_duke, torch.zeros_like(D_duke))
        sum_d_loss += d_loss.item()
        optimizerD.zero_grad()
        d_loss.backward()
        optimizerD.step()

        print('\r[epoch:' + str(epoch+1) + ', iter:' + str(current_num_iter)
        + '] CE: ' + str(round(sum_ce_loss/(sub_iter+1.0), 3)) 
        + ' | Acc: ' + str(round(100. * correct.item() / total, 2))
        + ' | LS: ' + str(round(sum_ls_loss/(sub_iter+1.), 3))
        + ' | GS: ' + str(round(sum_gs_loss/(sub_iter+1.), 3))
        + ' | A: ' + str(round(sum_a_loss/(sub_iter+1.), 3))
        + ' | D: ' + str(round(sum_d_loss/(sub_iter+1), 3)), end='****')

    myprint('\n[epoch:' + str(epoch+1) + ', iter:' + str(current_num_iter)
        + '] CE: ' + str(round(sum_ce_loss/(sub_iter+1.0), 3)) 
        + ' | Acc: ' + str(round(100. * correct.item() / total, 2))
        + ' | LS: ' + str(round(sum_ls_loss/(sub_iter+1.), 3))
        + ' | GS: ' + str(round(sum_gs_loss/(sub_iter+1.), 3))
        + ' | A: ' + str(round(sum_a_loss/(sub_iter+1.), 3))
        + ' | D: ' + str(round(sum_d_loss/(sub_iter+1), 3)), fid_log)
        
    if ((epoch+1)%1 == 0) or (epoch == (EPOCH-1)):
        myprint('\nEpoch: ' + str(epoch+1) + ' Testing...', fid_log)
        with torch.no_grad():
            this_duke_performance = SearchDuke(net, MY_GPU[0], [1])
            this_duke_r1, this_duke_map = this_duke_performance[0], this_duke_performance[-1]
            myprint('!!!!!!!!!!------>>>>>>>Duke R1:' + str(round(this_duke_r1,3)) + ' mAP:' + str(round(this_duke_map, 3)), fid_log)

            myprint('Cost ' + str(datetime.datetime.now() - BEGIN_TIME), fid_log)
            if this_duke_r1 > max_duke_r1:
                max_duke_r1, max_duke_map = this_duke_r1, this_duke_map
                max_duke_r5, max_duke_r10 = this_duke_performance[1], this_duke_performance[2]
                max_epoch = epoch+1

            myprint('Saving model: ' + SNAPSHOT +str(epoch+1).zfill(3)+'.pt', fid_log)
            torch.save(net.to('cpu').module.state_dict(), SNAPSHOT + str(epoch+1).zfill(3) + '.pt')
            net = net.to(device)
            myprint('------------------------------------------\n--->>> '
                + 'Max Duke R1:' + str(round(max_duke_r1, 3)) 
                + ' R5:' + str(round(max_duke_r5, 3))
                + ' R10:' + str(round(max_duke_r10, 3))
                + ' mAP:' + str(round(max_duke_map, 3))
                + ' Epoch:' + str(max_epoch)
                + '<<<---\n------------------------------------------', fid_log)

myprint('Cost ' + str(datetime.datetime.now() - BEGIN_TIME), fid_log)
myprint(str(datetime.datetime.now()), fid_log)
fid_log.close()
