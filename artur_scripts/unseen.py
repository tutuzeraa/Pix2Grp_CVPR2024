import torch, json, pprint

triplets = torch.load('cache/vg/vg_motif_anno/zeroshot_triplet.pytorch')
print(len(triplets), triplets[:5])          

#with open('categories_dict.json') as f:     
 #   cats = json.load(f)
#id2pred = {v['predicate_id']: v['predicate_name'] for v in cats['predicate']}
#print([id2pred[i] for i in zs_predicate[:10]])