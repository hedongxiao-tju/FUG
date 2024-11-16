#!/usr/bin/env python
# coding: utf-8

# In[17]:


from torch_geometric.datasets import Planetoid, CitationFull, Amazon, Coauthor, WikiCS, ppi
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Amazon, Coauthor, WebKB, Actor, WikipediaNetwork


def load_dataset(dataset_name, dataset_dir):

    print('Dataloader: Loading Dataset', dataset_name)
    assert dataset_name in ['Cora', 'CiteSeer', 'PubMed',
                            'dblp', 'Photo','Computers', 
                            'CS','Physics', 
                            'ogbn-products', 'ogbn-arxiv', 'Wiki','ppi',
                           'Cornell', 'Texas', 'Wisconsin',
                           'chameleon', 'crocodile', 'squirrel']
    
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name=dataset_name, 
                            transform=T.NormalizeFeatures())
        
    elif dataset_name == 'dblp':
        dataset = CitationFull(dataset_dir, name=dataset_name, 
                               transform=T.NormalizeFeatures()
                              )
        
    elif dataset_name in ['Photo','Computers']:
        dataset = Amazon(dataset_dir, name=dataset_name, 
                         transform=T.NormalizeFeatures())
        
    elif dataset_name in ['CS','Physics']:
        dataset = Coauthor(dataset_dir, name=dataset_name, 
                           transform=T.NormalizeFeatures())
        
    elif dataset_name in ['Wiki']:
        dataset = WikiCS(dataset_dir)#,
                                         #transform=T.NormalizeFeatures())
    elif dataset_name in ['ppi']:
        train = ppi.PPI(root = dataset_dir, transform=T.NormalizeFeatures(), split = 'train')
        val = ppi.PPI(root = dataset_dir, transform=T.NormalizeFeatures(), split = 'val')
        test = ppi.PPI(root = dataset_dir, transform=T.NormalizeFeatures(), split = 'test')
        dataset = [train, val, test]   
        
    elif dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
            return WebKB(
            dataset_dir,
            dataset_name,
            transform=T.NormalizeFeatures())
    elif dataset_name in ['chameleon', 'crocodile', 'squirrel']:
            return WikipediaNetwork(
            dataset_dir,
            dataset_name,
            transform=T.NormalizeFeatures())
    
    #print(dataset)
    print('Dataloader: Loading success.')
    print(dataset[0])
    
    return dataset


# In[ ]:




