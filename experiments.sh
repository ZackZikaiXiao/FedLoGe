Code for imanenet:
https://drive.google.com/file/d/1HLqF_n1Z8VUCIV3jDHhLmQXFrqs3CacR/view?usp=sharing

Code for inaturalist:
https://drive.google.com/file/d/1u5XF0AsDm0GRoEp3HJPZ_sSvvWb4BVCv/view?usp=sharing

nohup python fedloge.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 0 --num_users 40 --frac 0.3 > sse_c.log 2>&1 &
3409705

global model training:
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 2 > fedrod_repeat1.log 2>&1 &
只把classifier改成了etf
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 3 > fedrod_etf.log 2>&1 &
2163606

personalized model training:
nohup python load_and_infer.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 0 > load_and_infer.log 2>&1 &


测试etf：
nohup python fedrod_etf.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 0 > etf.log 2>&1 &
2113115

features后接g_aux
nohup python fedrod_etf.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 1 > etf_1.log 2>&1 &
2113269

加了个projection layer
nohup python fedrod_etf.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 2 > etf_projec_layer.log 2>&1 &
2113396


尝试一下sparse和其他的init结合
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 0 > kaiming_uniform_spar.log 2>&1 &
2212174
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 3 > kaiming_norm_spar.log 2>&1 &
2212246
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 0 > xavier_spar.log 2>&1 &
2212325
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 1 > uniform_spar.log 2>&1 &
2212423
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 2 > gassian_spar.log 2>&1 &
2212513
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 2 > orthogonal_spar.log 2>&1 &
2212624

nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 4 > kaiming_uniform.log 2>&1 &
2214640
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 4 > kaiming_norm.log 2>&1 &
2213604
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 5 > xavier.log 2>&1 &
2213494
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 5 > uniform.log 2>&1 &
2213371
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 6 > gassian.log 2>&1 &
2213255
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 6 > orthogonal.log 2>&1 &
2213147

nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 7 > default_nospar.log 2>&1 &
2213818
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 7 > default_spar.log 2>&1 &
2214052

nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 3 > etf_spar.log 2>&1 &
2243334


目前来看etf spar效果竟然是最好的。
下面要将features也约束到etf，首先加一个norm，然后来做内积方差极小化。
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 4 > etf_spar_featNorm.log 2>&1 &
2354658
学的太慢了，lr改成了0.3，epoch=1000
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 5 > etf_spar_featNorm_1.log 2>&1 &
2383230
怀疑instance level的norm约束不稳定，我在前面加一个projection layer把
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 1 > etf_spar_featNorm_projection.log 2>&1 &
2386528
直接fintune吧
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 3 > etf_spar_featNorm_finetune.log 2>&1 &
2378799
# nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 0 > kaiming_spar.log 2>&1 &
# nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 0 > kaiming_spar.log 2>&1 &
# nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 0 > kaiming_spar.log 2>&1 &


nohup python main.py --alpha_dirichlet 0.5 --IF 1 --beta 0 --gpu 1 > main.log 2>&1 &
2279153


预训练了一个cls
nohup python centra_demo.py > demo.log 2>&1 &
2325366
用这个fixed cls去train from scratch for backbone
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 0 > pretrain_cls.log 2>&1 &
2434027
性能只有0.38左右，看来cls的初始化确实很重要

dropout_ETF
每个minibatch随机一批节点
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 0 > dropout_ETF_batch.log 2>&1 &
效果不好，直接删了

每个round随机一批activation，每个sample的activation mask一样
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 4 > dropout_ETF_round.log 2>&1 &
2586465
每10个round随机一批activation，每个sample的activation mask一样
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 3 > dropout_ETF_10round.log 2>&1 &
2586578


每个round随机一批activation，每个sample的activation mask一样
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 4 > dropout_ETF_round.log 2>&1 &
2586465
每10个round随机一批activation，每个sample的activation mask一样
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 3 > dropout_ETF_10round.log 2>&1 &
2586578

每个round随机一批activation，每个sample的activation mask不一样
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 6 > official_dropout_ETF_round.log 2>&1 &
2594732
每10个round随机一批activation，每个sample的activation mask不一样
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 7 > official_dropout_ETF_10round.log 2>&1 &
2594842

每个round随机一批cls权重
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 5 > w_dropout_ETF_round.log 2>&1 &
2588472
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 6 > w_dropout_ETF_10round.log 2>&1 &
2588984

dropout这些统统效果不好。
我先spar预训练，再用pick up吧：
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 3 > spar_pick_up_1round.log 2>&1 &
效果不好

我要让不同类别feature相似度变大了。在pretrain model基础上，进一步train the backbone
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 3 > cos_ETF.log 2>&1 &
2762603

nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 4 > cos_push_pull_ETF.log 2>&1 &
2762678

约束feature的norm相等
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 4 > feats_eqe_norm_ETF.log 2>&1 &
2764228

约束feature的norm相等+相似度
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 4 > cos_norm_ETF.log 2>&1 &
2764088


spar etf + etf_projec_layer
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 0 > spar_etf_proj.log 2>&1 &
2898218

spar etf + PR dropout
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 3 > spar_etf_PRdot.log 2>&1 &
2906197

etf + PR dropout
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 3 > etf_PRdot.log 2>&1 &
2909571


用MMA regulation?

用balanced softmax
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 2 > spar_etf_bsm.log 2>&1 &
3041272

用norm估计的balanced softmax
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 0 > spar_etf_bsm_norm_est.log 2>&1 &
3053433

用构造的sparse etf进行训练,norm=1
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 0 > my_designed_sparse_etf.log 2>&1 &
3499238
用构造的sparse etf进行训练,norm=0.1
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 1 > my_designed_sparse_etf_norm01.log 2>&1 &
3504052
用构造的sparse etf进行训练,norm=0.03
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 1 > my_designed_sparse_etf_norm003.log 2>&1 &
3504741
用构造的sparse etf进行训练,norm=0.15
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 2 > my_designed_sparse_etf_norm015.log 2>&1 &
3505098

不sparse，直接用norm=0.15的etf来
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 3 > etf_norm015.log 2>&1 &
3508021

if cls_switch == "ETF":
    # 初始化ETF分类器 
    etf = ETF_Classifier(in_features, out_features) 
    # 新建线性层,权重使用ETF分类器的ori_M
    g_head = nn.Linear(in_features, out_features).to(args.device) 
    # sparse_etf_mat = etf.gen_sparse_ETF()
    g_head.weight.data = etf.ori_M.to(args.device)
    # g_head.weight.data = sparse_etf_mat.to(args.device)
    g_head.weight.data = g_head.weight.data.t()
    g_head.weight.data *= 0.15


cifar 10的spar
以下是用norm的balanced softmax搞出来的
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 7 > cifar10_nospar.log 2>&1 &
3112359
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 2 > cifar10_nospar_2.log 2>&1 &
3132781
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 7 > cifar10_spar10.log 2>&1 &
3112670
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 7 > cifar10_spar20.log 2>&1 &
3112783
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 6 > cifar10_spar30.log 2>&1 &
3112924
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 6 > cifar10_spar40.log 2>&1 &
3113064
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 6 > cifar10_spar50.log 2>&1 &
3113395
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 5 > cifar10_spar60.log 2>&1 &
3113511
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 5 > cifar10_spar70.log 2>&1 &
3113647
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 5 > cifar10_spar80.log 2>&1 &
3113773
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 4 > cifar10_spar90.log 2>&1 &
3113894





保存nospar的etf的性能，为了后面做分析用
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 1 > cifar100_nospar_save.log 2>&1 &
[1] 3129972


用self search得到的稀疏化继续finetune训练
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 2 > self_search_spar.log 2>&1 &
3277242




nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 2 > self_search_spar.log 2>&1 &


###################################################################################################


FedAvg
nohup python main.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 0 > a_fedavg.log 2>&1 &

nohup python main.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 0 > b_fedavg.log 2>&1 &

nohup python main.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 0 > c_fedavg.log 2>&1 &

nohup python main.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 1 > d_fedavg.log 2>&1 &

nohup python main.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 1 > e_fedavg.log 2>&1 &

nohup python main.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 1 > f_fedavg.log 2>&1 &

nohup python main.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 2 > g_fedavg.log 2>&1 &

nohup python main.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 2 > h_fedavg.log 2>&1 &

更新model
nohup python main.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 0 > a_fedavg_tmp.log 2>&1 &

nohup python main.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 0 > b_fedavg_tmp.log 2>&1 &

nohup python main.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 3 > c_fedavg_tmp.log 2>&1 &

nohup python main.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 1 > d_fedavg_tmp.log 2>&1 &

nohup python main.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 1 > e_fedavg_tmp.log 2>&1 &

nohup python main.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 3 > f_fedavg_tmp.log 2>&1 &

nohup python main.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 2 > g_fedavg_tmp.log 2>&1 &

nohup python main.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 2 > h_fedavg_tmp.log 2>&1 &


FedProx
nohup python main.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0.01 --gpu 0 > a_fedprox.log 2>&1 &

nohup python main.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0.01 --gpu 0 > b_fedprox.log 2>&1 &
 
nohup python main.py --alpha_dirichlet 1 --IF 0.02 --beta 0.01 --gpu 3 > c_fedprox.log 2>&1 &

nohup python main.py --alpha_dirichlet 1 --IF 0.01 --beta 0.01 --gpu 3 > d_fedprox.log 2>&1 &

nohup python main.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0.01 --gpu 4 > e_fedprox.log 2>&1 &

nohup python main.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0.01 --gpu 4 > f_fedprox.log 2>&1 &

nohup python main.py --alpha_dirichlet 1 --IF 0.02 --beta 0.01 --gpu 5 > g_fedprox.log 2>&1 &

nohup python main.py --alpha_dirichlet 1 --IF 0.01 --beta 0.01 --gpu 5 > h_fedprox.log 2>&1 &



FedBN
/home/zikaixiao/zikai/aapfl/fl_gba_cifar100/fedbn.py
nohup python fedbn.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 6 > a_fedbn.log 2>&1 &

nohup python fedbn.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 6 > b_fedbn.log 2>&1 &

nohup python fedbn.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 7 > c_fedbn.log 2>&1 &

nohup python fedbn.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 7 > d_fedbn.log 2>&1 &

nohup python fedbn.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 1 > e_fedbn.log 2>&1 &

nohup python fedbn.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 1 > f_fedbn.log 2>&1 &

nohup python fedbn.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 2 > g_fedbn.log 2>&1 &

nohup python fedbn.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 2 > h_fedbn.log 2>&1 &





FedROD
/home/zikaixiao/zikai/aapfl/pfedlt_cifar100_fedrod/fedrod.py
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 0 > a_fedbn.log 2>&1 &
3373133
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 0 > b_fedbn.log 2>&1 &
3373134
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 1 > c_fedbn.log 2>&1 &
3373135
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 1 > d_fedbn.log 2>&1 &
3373136
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 2 > e_fedbn.log 2>&1 &
3373137
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 2 > f_fedbn.log 2>&1 &
3373138
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 3 > g_fedbn.log 2>&1 &
3373139
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 3 > h_fedbn.log 2>&1 &
3373368


FedPer
/home/zikaixiao/zikai/aapfl/fl_gba_cifar100/fedper.py
nohup python fedper.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 4 > a_fedper.log 2>&1 &

nohup python fedper.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 4 > b_fedper.log 2>&1 &

nohup python fedper.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 5 > c_fedper.log 2>&1 &

nohup python fedper.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 5 > d_fedper.log 2>&1 &

nohup python fedper.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 6 > e_fedper.log 2>&1 &

nohup python fedper.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 6 > f_fedper.log 2>&1 &

nohup python fedper.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 7 > g_fedper.log 2>&1 &

nohup python fedper.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 7 > h_fedper.log 2>&1 &

FedRep
/home/zikaixiao/zikai/aapfl/fl_gba_cifar100/fedpep.py
nohup python fedrep.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 0 > a_fedrep.log 2>&1 &
3707138
nohup python fedrep.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 0 > b_fedrep.log 2>&1 &
3707226
nohup python fedrep.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 1 > c_fedrep.log 2>&1 &
3707313
nohup python fedrep.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 1 > d_fedrep.log 2>&1 &
3707390
nohup python fedrep.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 2 > e_fedrep.log 2>&1 &
3707483
nohup python fedrep.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 2 > f_fedrep.log 2>&1 &
3707575
nohup python fedrep.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 3 > g_fedrep.log 2>&1 &
3707679
nohup python fedrep.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 3 > h_fedrep.log 2>&1 &
3707744

FedETF
/home/zikaixiao/zikai/aaPFL/pfedlt/fedrod_etf.py
nohup python fedrod_etf.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 4 > a_fedetf.log 2>&1 &
3749078
nohup python fedrod_etf.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 4 > b_fedetf.log 2>&1 &
3749217
nohup python fedrod_etf.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 5 > c_fedetf.log 2>&1 &
3749342
nohup python fedrod_etf.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 5 > d_fedetf.log 2>&1 &
3749462
nohup python fedrod_etf.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 6 > e_fedetf.log 2>&1 &

nohup python fedrod_etf.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 6 > f_fedetf.log 2>&1 &

nohup python fedrod_etf.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 7 > g_fedetf.log 2>&1 &

nohup python fedrod_etf.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 7 > h_fedetf.log 2>&1 &



ratio loss
/home/zikaixiao/zikai/aaPFL/pfedlt/ratio_loss.py
nohup python ratio_loss.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 4 > a_ratio_loss.log 2>&1 &
3727489
nohup python ratio_loss.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 4 > b_ratio_loss.log 2>&1 &
3727611
nohup python ratio_loss.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 5 > c_ratio_loss.log 2>&1 &
3727737
nohup python ratio_loss.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 5 > d_ratio_loss.log 2>&1 &
3727853
nohup python ratio_loss.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 6 > e_ratio_loss.log 2>&1 &
3728002
nohup python ratio_loss.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 6 > f_ratio_loss.log 2>&1 &
3728124
nohup python ratio_loss.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 7 > g_ratio_loss.log 2>&1 &
3728266
nohup python ratio_loss.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 7 > h_ratio_loss.log 2>&1 &
3728469


ditto
/home/zikaixiao/zikai/aapfl/pfed_lastest/ditto.py
nohup python ditto.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 4 > a_ditto.log 2>&1 &
3711778
nohup python ditto.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 4 > b_ditto.log 2>&1 &
3711915
nohup python ditto.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 5 > c_ditto.log 2>&1 &
3712050
nohup python ditto.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 5 > d_ditto.log 2>&1 &
3712181
nohup python ditto.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 6 > e_ditto.log 2>&1 &
3712324
nohup python ditto.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 6 > f_ditto.log 2>&1 &
3712476
nohup python ditto.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 7 > g_ditto.log 2>&1 &
3712609
nohup python ditto.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 7 > h_ditto.log 2>&1 &
3712801


Ours
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 0 > a_ours.log 2>&1 &
10281
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 0 > b_ours.log 2>&1 &
10625
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 1 > c_ours.log 2>&1 &
10941
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 1 > d_ours.log 2>&1 &
11246
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 2 > e_ours.log 2>&1 &
11560
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 2 > f_ours.log 2>&1 &
11601
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.02 --beta 0 --gpu 3 > g_ours.log 2>&1 &
12180
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 3 > h_ours.log 2>&1 &
12499


测试不同的norm
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 0 > norm0_02.log 2>&1 &
391283
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 0 > norm0_04.log 2>&1 &
391397
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 1 > norm0_06.log 2>&1 &
391528
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 1 > norm0_08.log 2>&1 &
391649
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 2 > norm0_1.log 2>&1 &
391785
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 2 > norm0_2.log 2>&1 &
391910
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 3 > norm0_4.log 2>&1 &
392043
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 3 > norm0_6.log 2>&1 &
392168
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 4 > norm0_8.log 2>&1 &
392300
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 4 > norm1.log 2>&1 &
392433
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 5 > norm2.log 2>&1 &
392658
nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 5 > norm4.log 2>&1 &

nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 6 > norm6.log 2>&1 &

nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 6 > norm8.log 2>&1 &

nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 7 > norm10.log 2>&1 &

nohup python fedrod.py --alpha_dirichlet 1 --IF 0.01 --beta 0 --gpu 7 > norm0_01.log 2>&1 &
391135




Rebuttal

FedAvg
nohup python main.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 5 --num_users 40 --frac 0.3 > rebu_fedavg_40_30.log 2>&1 &
nohup python main.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 5 --num_users 100 --frac 1 > rebu_fedavg_100_100.log 2>&1 &
nohup python main.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 7 --num_users 100 --frac 0.3 > rebu_fedavg_100_30.log 2>&1 &



FedLoGe
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 4 --num_users 40 --frac 0.3 > rebu_fedloge_40_30.log 2>&1 &
1298768
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 7 --num_users 100 --frac 0.3 > rebu_fedloge_100_30.log 2>&1 &



FedROD
/home/zikaixiao/zikai/aapfl/pfedlt_cifar100_fedrod/fedrod.py
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 7 --num_users 40 --frac 0.3 > rebu_fedrod_40_30.log 2>&1 &

nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 5 --num_users 100 --frac 0.3 > rebu_fedrod_100_30.log 2>&1 &


FedETF
nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.02 --beta 0 --gpu 5 --num_users 40 --frac 0.3 > rebu_fedetf_40_30.log 2>&1 &

nohup python fedrod.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 5 --num_users 100 --frac 0.3 > rebu_fedetf_100_30.log 2>&1 &




New Clients
python new_clients.py --alpha_dirichlet 0.5 --IF 0.01 --beta 0 --gpu 7 --num_users 20 --frac 1 --local_ep 1
