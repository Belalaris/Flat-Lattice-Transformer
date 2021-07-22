import sys
sys.path.append('../')
from load_data import *
import argparse
from paths import *
from V0.models import Lattice_Transformer_SeqLabel, Transformer_SeqLabel
import torch
import collections
import torch.nn as nn
from V0.add_lattice import equip_chinese_ner_with_lexicon
from utils import print_info

from fastNLP.core.metrics import _bmes_tag_to_spans, _bioes_tag_to_spans, \
    _bio_tag_to_spans, _bmeso_tag_to_spans
from fastNLP.core.predictor import Predictor


load_dataset_seed = 100

parser = argparse.ArgumentParser()

parser.add_argument('--train_clip',default=False,help='是不是要把train的char长度限制在200以内')
parser.add_argument('--device', default='0')
parser.add_argument('--debug', default=0,type=int)
parser.add_argument('--gpumm',default=False,help='查看显存')
parser.add_argument('--see_param',default=False)
parser.add_argument('--seed', default=11242019,type=int)
parser.add_argument('--number_normalized',type=int,default=0,
                    choices=[0,1,2,3],help='0不norm，1只norm char,2norm char和bigram，3norm char，bigram和lattice')
parser.add_argument('--lexicon_name',default='yj',choices=['lk','yj'])
parser.add_argument('--update_every',default=1,type=int)
parser.add_argument('--use_pytorch_dropout',type=int,default=0)

parser.add_argument('--char_min_freq',default=1,type=int)
parser.add_argument('--bigram_min_freq',default=1,type=int)
parser.add_argument('--lattice_min_freq',default=1,type=int)
parser.add_argument('--only_train_min_freq',default=True)
parser.add_argument('--only_lexicon_in_train',default=False)

parser.add_argument('--word_min_freq',default=1,type=int)

# hyper of training
# parser.add_argument('--early_stop',default=40,type=int)
parser.add_argument('--init',default='uniform',help='norm|uniform')
parser.add_argument('--self_supervised',default=False)
parser.add_argument('--norm_embed',default=True)
parser.add_argument('--norm_lattice_embed',default=True)

# hyper of model
parser.add_argument('--model',default='transformer',help='lstm|transformer')
parser.add_argument('--lattice',default=1,type=int)
parser.add_argument('--use_bigram', default=1,type=int)
parser.add_argument('--hidden', default=-1,type=int)
parser.add_argument('--ff', default=3,type=int)
parser.add_argument('--layer', default=1,type=int)
parser.add_argument('--head', default=8,type=int)
parser.add_argument('--head_dim',default=20,type=int)
parser.add_argument('--scaled',default=False)
parser.add_argument('--ff_activate',default='relu',help='leaky|relu')

parser.add_argument('--k_proj',default=False)
parser.add_argument('--q_proj',default=True)
parser.add_argument('--v_proj',default=True)
parser.add_argument('--r_proj',default=True)

parser.add_argument('--attn_ff',default=False)

# parser.add_argument('--rel_pos', default=False)
parser.add_argument('--use_abs_pos',default=False)
parser.add_argument('--use_rel_pos',default=True)
#相对位置和绝对位置不是对立的，可以同时使用
parser.add_argument('--rel_pos_shared',default=True)
parser.add_argument('--add_pos', default=False)
parser.add_argument('--learn_pos', default=False)
parser.add_argument('--pos_norm',default=False)
parser.add_argument('--rel_pos_init',default=1)
parser.add_argument('--four_pos_shared',default=True,help='只针对相对位置编码，指4个位置编码是不是共享权重')
parser.add_argument('--four_pos_fusion',default='ff_two',choices=['ff','attn','gate','ff_two','ff_linear'],
                    help='ff就是输入带非线性隐层的全连接，'
                         'attn就是先计算出对每个位置编码的加权，然后求加权和'
                         'gate和attn类似，只不过就是计算的加权多了一个维度')

parser.add_argument('--four_pos_fusion_shared',default=True,help='是不是要共享4个位置融合之后形成的pos')

# parser.add_argument('--rel_pos_scale',default=2,help='在lattice且用相对位置编码时，由于中间过程消耗显存过大，'
#                                                  '所以可以使4个位置的初始embedding size缩小，'
#                                                  '最后融合时回到正常的hidden size即可')

parser.add_argument('--pre', default='')
parser.add_argument('--post', default='an')

over_all_dropout =  -1
parser.add_argument('--embed_dropout_before_pos',default=False)
parser.add_argument('--embed_dropout', default=0.5,type=float)
parser.add_argument('--gaz_dropout',default=0.5,type=float)
parser.add_argument('--output_dropout', default=0.3,type=float)
parser.add_argument('--pre_dropout', default=0.5,type=float)
parser.add_argument('--post_dropout', default=0.3,type=float)
parser.add_argument('--ff_dropout', default=0.15,type=float)
parser.add_argument('--ff_dropout_2', default=-1,type=float,help='FF第二层过完后的dropout，之前没管这个的时候是0')
parser.add_argument('--attn_dropout',default=0,type=float)
parser.add_argument('--embed_dropout_pos',default='0')
parser.add_argument('--abs_pos_fusion_func',default='nonlinear_add',
                    choices=['add','concat','nonlinear_concat','nonlinear_add','concat_nonlinear','add_nonlinear'])


parser.add_argument('--dataset', default='resume', help='weibo|resume|ontonote|msra|clue')

args = parser.parse_args()
if args.ff_dropout_2 < 0:
    args.ff_dropout_2 = args.ff_dropout

if over_all_dropout>0:
    args.embed_dropout = over_all_dropout
    args.output_dropout = over_all_dropout
    args.pre_dropout = over_all_dropout
    args.post_dropout = over_all_dropout
    args.ff_dropout = over_all_dropout
    args.attn_dropout = over_all_dropout


if args.lattice and args.use_rel_pos and args.update_every == 1:
    args.train_clip = True

if args.device!='cpu':
    assert args.device.isdigit()
    device = torch.device('cuda:{}'.format(args.device))
else:
    device = torch.device('cpu')

refresh_data = True
# import random
# print('**'*12,random.random,'**'*12)

cache_path = 'cache_' + args.dataset

# for k,v in args.__dict__.items():
#     print_info('{}:{}'.format(k,v))


raw_dataset_cache_name = os.path.join(cache_path,args.dataset+
                          '_trainClip#{}'.format(args.train_clip)
                          +'_bgminfreq#{}'.format(args.bigram_min_freq)
                          +'_char_min_freq#{}'.format(args.char_min_freq)
                          +'_word_min_freq#{}'.format(args.word_min_freq)
                          +'_only_train_min_freq#{}'.format(args.only_train_min_freq)
                          +'_number_norm#{}'.format(args.number_normalized)
                          +'_load_dataset_seed#{}'.format(load_dataset_seed)
                          )

# if args.dataset == 'ontonotes':
#     datasets,vocabs,embeddings = load_ontonotes4ner(ontonote4ner_cn_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
#                                                     _refresh=refresh_data,index_token=False,train_clip=args.train_clip,
#                                                     _cache_fp=raw_dataset_cache_name,
#                                                     char_min_freq=args.char_min_freq,
#                                                     bigram_min_freq=args.bigram_min_freq,
#                                                     only_train_min_freq=args.only_train_min_freq
#                                                     )
if args.dataset == 'resume':
    datasets,vocabs,embeddings = load_resume_ner(resume_ner_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
                                                    _refresh=refresh_data,index_token=False,
                                                 _cache_fp=raw_dataset_cache_name,
                                                 char_min_freq=args.char_min_freq,
                                                 bigram_min_freq=args.bigram_min_freq,
                                                 only_train_min_freq=args.only_train_min_freq
                                                    )
elif args.dataset == 'weibo':
    datasets,vocabs,embeddings = load_weibo_ner(weibo_ner_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
                                                    _refresh=refresh_data,index_token=False,
                                                _cache_fp=raw_dataset_cache_name,
                                                char_min_freq=args.char_min_freq,
                                                bigram_min_freq=args.bigram_min_freq,
                                                only_train_min_freq=args.only_train_min_freq
                                                    )
elif args.dataset == 'clue':
    datasets,vocabs,embeddings = load_clue_ner(clue_ner_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
                                                    _refresh=refresh_data,index_token=False,
                                                    _cache_fp=raw_dataset_cache_name,
                                                    char_min_freq=args.char_min_freq,
                                                    bigram_min_freq=args.bigram_min_freq,
                                                    only_train_min_freq=args.only_train_min_freq
                                                    )
# elif args.dataset == 'toy':
#     datasets,vocabs,embeddings = load_toy_ner(toy_ner_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
#                                                     _refresh=refresh_data,index_token=False,train_clip=args.train_clip,
#                                                     _cache_fp=raw_dataset_cache_name
#                                                     )


elif args.dataset == 'msra':
    datasets,vocabs,embeddings = load_msra_ner_1(msra_ner_cn_path,yangjie_rich_pretrain_unigram_path,
                                                           yangjie_rich_pretrain_bigram_path,
                                                           _refresh=refresh_data,index_token=False,train_clip=args.train_clip,
                                                           _cache_fp=raw_dataset_cache_name,
                                                           char_min_freq=args.char_min_freq,
                                                           bigram_min_freq=args.bigram_min_freq,
                                                           only_train_min_freq=args.only_train_min_freq
                                                           )

if args.gaz_dropout < 0:
    args.gaz_dropout = args.embed_dropout

args.hidden = args.head_dim * args.head
args.ff = args.hidden * args.ff

if args.dataset == 'weibo':
    args.ff_dropout = 0.3
    args.ff_dropout_2 = 0.3
    args.head_dim = 16
    args.ff = 384
    args.hidden = 128
    args.init = 'uniform'
    args.seed = 11741

elif args.dataset == 'resume':
    args.head_dim = 16
    args.ff = 384
    args.hidden = 128
    args.seed = 15460

elif args.dataset == 'ontonotes':
    args.seed = 17664
    args.update_every = 2
    pass

elif args.dataset == 'clue':
    args.seed = 17664
    args.update_every = 2
    pass

elif args.dataset == 'msra':
    pass



if args.lexicon_name == 'lk':
    yangjie_rich_pretrain_word_path = lk_word_path_2

print('用的词表的路径:{}'.format(yangjie_rich_pretrain_word_path))

w_list = load_yangjie_rich_pretrain_word_list(yangjie_rich_pretrain_word_path,
                                              _refresh=refresh_data,
                                              _cache_fp=cache_path+'/'+args.lexicon_name)

cache_name = os.path.join(cache_path,(args.dataset+'_lattice'+'_only_train#{}'+ '_trainClip#{}'+'_norm_num#{}'
                                   +'_char_min_freq#{}'+'_bigram_min_freq#{}'+'_word_min_freq#{}'+'_only_train_min_freq#{}'
                                   +'_number_norm#{}'+'_lexicon#{}'+'_load_dataset_seed#{}')
                          .format(args.only_lexicon_in_train,
                          args.train_clip,args.number_normalized,args.char_min_freq,
                          args.bigram_min_freq,args.word_min_freq,args.only_train_min_freq,
                          args.number_normalized,args.lexicon_name,load_dataset_seed))
datasets,vocabs,embeddings = equip_chinese_ner_with_lexicon(datasets,vocabs,embeddings,
                                                            w_list,yangjie_rich_pretrain_word_path,
                                                         _refresh=refresh_data,_cache_fp=cache_name,
                                                         only_lexicon_in_train=args.only_lexicon_in_train,
                                                            word_char_mix_embedding_path=yangjie_rich_pretrain_char_and_word_path,
                                                            number_normalized=args.number_normalized,
                                                            lattice_min_freq=args.lattice_min_freq,
                                                            only_train_min_freq=args.only_train_min_freq)

print('train:{}'.format(len(datasets['train'])))
avg_seq_len = 0
avg_lex_num = 0
avg_seq_lex = 0
train_seq_lex = []
dev_seq_lex = []
test_seq_lex = []
train_seq = []
dev_seq = []
test_seq = []
for k,v in datasets.items():
    max_seq_len = 0
    max_lex_num = 0
    max_seq_lex = 0
    max_seq_len_i = -1
    for i in range(len(v)):
        if max_seq_len < v[i]['seq_len']:
            max_seq_len = v[i]['seq_len']
            max_seq_len_i = i
        # max_seq_len = max(max_seq_len,v[i]['seq_len'])
        max_lex_num = max(max_lex_num,v[i]['lex_num'])
        max_seq_lex = max(max_seq_lex,v[i]['lex_num']+v[i]['seq_len'])

        avg_seq_len+=v[i]['seq_len']
        avg_lex_num+=v[i]['lex_num']
        avg_seq_lex+=(v[i]['seq_len']+v[i]['lex_num'])
        if k == 'train':
            train_seq_lex.append(v[i]['lex_num']+v[i]['seq_len'])
            train_seq.append(v[i]['seq_len'])
            # if v[i]['seq_len'] >200:
            #     print('train里这个句子char长度已经超了200了')
            #     print(''.join(list(map(lambda x:vocabs['char'].to_word(x),v[i]['chars']))))
            # else:
            #     if v[i]['seq_len']+v[i]['lex_num']>400:
            #         print('train里这个句子char长度没超200，但是总长度超了400')
            #         print(''.join(list(map(lambda x: vocabs['char'].to_word(x), v[i]['chars']))))
        if k == 'dev':
            dev_seq_lex.append(v[i]['lex_num']+v[i]['seq_len'])
            dev_seq.append(v[i]['seq_len'])
        if k == 'test':
            test_seq_lex.append(v[i]['lex_num']+v[i]['seq_len'])
            test_seq.append(v[i]['seq_len'])


    # print('{} 最长的句子是:{}'.format(k,list(map(lambda x:vocabs['char'].to_word(x),v[max_seq_len_i]['chars']))))
    print('{} max_seq_len:{}'.format(k,max_seq_len))
    print('{} max_lex_num:{}'.format(k, max_lex_num))
    print('{} max_seq_lex:{}'.format(k, max_seq_lex))


max_seq_len = max(* map(lambda x:max(x['seq_len']),datasets.values()))

show_index = 4
print('raw_chars:{}'.format(list(datasets['train'][show_index]['raw_chars'])))
print('lexicons:{}'.format(list(datasets['train'][show_index]['lexicons'])))
print('lattice:{}'.format(list(datasets['train'][show_index]['lattice'])))
print('raw_lattice:{}'.format(list(map(lambda x:vocabs['lattice'].to_word(x),
                                  list(datasets['train'][show_index]['lattice'])))))
print('lex_s:{}'.format(list(datasets['train'][show_index]['lex_s'])))
print('lex_e:{}'.format(list(datasets['train'][show_index]['lex_e'])))
print('pos_s:{}'.format(list(datasets['train'][show_index]['pos_s'])))
print('pos_e:{}'.format(list(datasets['train'][show_index]['pos_e'])))


for k, v in datasets.items():
    if args.lattice:
        v.set_input('lattice','bigrams','seq_len','target')
        v.set_input('lex_num','pos_s','pos_e')
        v.set_target('target','seq_len')
    else:
        v.set_input('chars','bigrams','seq_len','target')
        v.set_target('target', 'seq_len')

from utils import norm_static_embedding
# print(embeddings['char'].embedding.weight[:10])
if args.norm_embed>0:
    print('embedding:{}'.format(embeddings['char'].embedding.weight.size()))
    print('norm embedding')
    for k,v in embeddings.items():
        norm_static_embedding(v,args.norm_embed)

if args.norm_lattice_embed>0:
    print('embedding:{}'.format(embeddings['lattice'].embedding.weight.size()))
    print('norm lattice embedding')
    for k,v in embeddings.items():
        norm_static_embedding(v,args.norm_embed)

mode = {}
mode['debug'] = args.debug
mode['gpumm'] = args.gpumm
dropout = collections.defaultdict(int)
dropout['embed'] = args.embed_dropout
dropout['gaz'] = args.gaz_dropout
dropout['output'] = args.output_dropout
dropout['pre'] = args.pre_dropout
dropout['post'] = args.post_dropout
dropout['ff'] = args.ff_dropout
dropout['ff_2'] = args.ff_dropout_2
dropout['attn'] = args.attn_dropout


# torch.backends.cudnn.benchmark = False
# fitlog.set_rng_seed(args.seed)
# torch.backends.cudnn.benchmark = False

if args.model == 'transformer':
    if args.lattice:
        model = Lattice_Transformer_SeqLabel(embeddings['lattice'], embeddings['bigram'], args.hidden, len(vocabs['label']),
                                     args.head, args.layer, args.use_abs_pos,args.use_rel_pos,
                                     args.learn_pos, args.add_pos,
                                     args.pre, args.post, args.ff, args.scaled,dropout,args.use_bigram,
                                     mode,device,vocabs,
                                     max_seq_len=max_seq_len,
                                     rel_pos_shared=args.rel_pos_shared,
                                     k_proj=args.k_proj,
                                     q_proj=args.q_proj,
                                     v_proj=args.v_proj,
                                     r_proj=args.r_proj,
                                     self_supervised=args.self_supervised,
                                     attn_ff=args.attn_ff,
                                     pos_norm=args.pos_norm,
                                     ff_activate=args.ff_activate,
                                     abs_pos_fusion_func=args.abs_pos_fusion_func,
                                     embed_dropout_pos=args.embed_dropout_pos,
                                     four_pos_shared=args.four_pos_shared,
                                     four_pos_fusion=args.four_pos_fusion,
                                     four_pos_fusion_shared=args.four_pos_fusion_shared,
                                     use_pytorch_dropout=args.use_pytorch_dropout
                                     )
    else:
        model = Transformer_SeqLabel(embeddings['lattice'], embeddings['bigram'], args.hidden, len(vocabs['label']),
                                     args.head, args.layer, args.use_abs_pos,args.use_rel_pos,
                                     args.learn_pos, args.add_pos,
                                     args.pre, args.post, args.ff, args.scaled,dropout,args.use_bigram,
                                     mode,device,vocabs,
                                     max_seq_len=max_seq_len,
                                     rel_pos_shared=args.rel_pos_shared,
                                     k_proj=args.k_proj,
                                     q_proj=args.q_proj,
                                     v_proj=args.v_proj,
                                     r_proj=args.r_proj,
                                     self_supervised=args.self_supervised,
                                     attn_ff=args.attn_ff,
                                     pos_norm=args.pos_norm,
                                     ff_activate=args.ff_activate,
                                     abs_pos_fusion_func=args.abs_pos_fusion_func,
                                     embed_dropout_pos=args.embed_dropout_pos
                                     )

    # print(Transformer_SeqLabel.encoder.)

# elif args.model =='lstm':
#     model = LSTM_SeqLabel_True(embeddings['char'],embeddings['bigram'],embeddings['bigram'],args.hidden,
#                                len(vocabs['label']),
#                           bidirectional=True,device=device,
#                           embed_dropout=args.embed_dropout,output_dropout=args.output_dropout,use_bigram=True,
#                           debug=args.debug)

# for n,p in model.named_parameters():
#     print('{}:{}'.format(n,p.size()))


with torch.no_grad():
    print_info('{}init pram{}'.format('*'*15,'*'*15))
    for n,p in model.named_parameters():
        if 'embedding' not in n and 'pos' not in n and 'pe' not in n \
                and 'bias' not in n and 'crf' not in n and p.dim()>1:
            try:
                if args.init == 'uniform':
                    nn.init.xavier_uniform_(p)
                    print_info('xavier uniform init:{}'.format(n))
                elif args.init == 'norm':
                    print_info('xavier norm init:{}'.format(n))
                    nn.init.xavier_normal_(p)
            except:
                print_info(n)
                exit(1208)
    print_info('{}init pram{}'.format('*' * 15, '*' * 15))

encoding_type = 'bmeso'
if args.dataset == 'msra':
    encoding_type == 'bioes'
elif args.dataset == 'weibo':
    encoding_type == 'bio'
elif args.dataset == 'clue':
    encoding_type = 'bio'


if args.see_param:
    for n,p in model.named_parameters():
        print_info('{}:{}'.format(n,p.size()))
    print_info('see_param mode: finish')
    if not args.debug:
        exit(1208)


if 'msra' in args.dataset :
    datasets['dev']  = datasets['test']

# print('parameter weight:')
# print(model.state_dict()['encoder.layer_0.attn.w_q.weight'])




def predictor_load_weigh(model, model_path):
    # model test|predict
    states = torch.load(model_path)
    model.load_state_dict(states) # 这里的model是加载权重之后的model
    predictor = Predictor(model)
    return predictor


def predict(predictor, set='test', index=0, encoding_type='bmeso'):
    """
    :param set: test|dev|train
    :param index: index of data item
    :param type: bmeso|bioes|bio|bmes
    :return: list of tuple
    :rtype: list
    """
    raw_chars = datasets[set][index]['raw_chars']  # 原始数据
    label_array = predictor.predict(datasets[set][index:index+1])['pred'][0]  # 预测结果
    label_list = list(label_array)[0]

    assert len(label_list.shape) == 1
    assert len(label_list) == len(raw_chars)
    print("raw_chars: \n", raw_chars)

    tag_list = [vocabs['label'].to_word(i) for i in label_list]
    spans = []
    if encoding_type == 'bmes':
        spans = _bmes_tag_to_spans(tag_list, ignore_labels=None)
    elif encoding_type == 'bio':
        spans = _bio_tag_to_spans(tag_list, ignore_labels=None)
    elif encoding_type == 'bmeso':
        spans = _bmeso_tag_to_spans(tag_list, ignore_labels=None)
    elif encoding_type == 'bioes':
        spans = _bioes_tag_to_spans(tag_list, ignore_labels=None)

    named_entities = []
    for span in spans:
        tag_slice_begin, tag_slice_end = span[1][0], span[1][1]
        entity_name_list = raw_chars[tag_slice_begin: tag_slice_end]
        entity_name = ''.join(entity_name_list)
        named_entities.append((entity_name, span[0]))

    return named_entities

model_path = '../model/resume_V0_2021_07_21_10_39_22.pkl'

predictor = predictor_load_weigh(model, model_path)
prediction = predict(predictor, set='dev', index=1)
print("named entity: \n", prediction)

'''
raw_chars: 
 ['历', '任', '公', '司', '副', '总', '经', '理', '、', '总', '工', '程', '师', '，']
named entity: 
 [('公司', 'org'), ('副总经理', 'title'), ('总工程师', 'title')]
'''