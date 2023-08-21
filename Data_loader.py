import torch
import dgl

def Get_RecipeGraph(path):
    print('Generating Recipe Graph:')
    #Get recipe and ingredient information
    edge_src, edge_dst, r_i_edge_weight = torch.load(path + '/recipe_ingredient_edges.pt')

    #Get ingredient and compound information
    f = open('data/C_list.txt', 'r')
    C_list = f.read()
    C_list = eval(C_list)
    f.close()
    f = open('data/I_list.txt', 'r')
    I_list = f.read()
    I_list = eval(I_list)
    f.close()

    # Get recipe and ingredient
    recipe_edge_src, recipe_edge_dst, recipe_edge_weight = torch.load(
        path + '/recipe_edges.pt')
    ingre_edge_src, ingre_edge_dst, ingre_edge_weight = torch.load(
        path + '/ingredient_edges.pt')

    graph = dgl.heterograph({
        ('compound', 'c-i', 'ingredient'): (C_list, I_list),
        ('ingredient', 'i-c', 'compound'): (I_list, C_list),
        ('recipe', 'r-i', 'ingredient'): (edge_src, edge_dst),
        ('ingredient', 'i-r', 'recipe'): (edge_dst, edge_src),
        ('recipe', 'r-r', 'recipe'): (recipe_edge_src, recipe_edge_dst),
        ('ingredient', 'i-i', 'ingredient'): (ingre_edge_src, ingre_edge_dst)
    })

    graph.edges['r-i'].data['weight'] = torch.FloatTensor(r_i_edge_weight)
    graph.edges['i-r'].data['weight'] = torch.FloatTensor(r_i_edge_weight)
    graph.edges['r-r'].data['weight'] = torch.FloatTensor(recipe_edge_weight)
    graph.edges['i-i'].data['weight'] = torch.FloatTensor(ingre_edge_weight)

    #Get node features
    recipe_instruction_features = torch.load(path + 'recipe_instruction_features.pt')
    compound_features = torch.load(path + 'compound_features.pth')
    ingredient_features = torch.load(path + '/ingredient_features.pt')

    train_mask = torch.load(path + '/recipe_nodes_train_mask.pt')
    val_mask = torch.load(path + '/recipe_nodes_val_mask.pt')
    test_mask = torch.load(path + '/recipe_nodes_test_mask.pt')
    recipe_nodes_labels = torch.load(path + '/recipe_nodes_labels.pt')



    graph.nodes['compound'].data['com_feature'] = compound_features
    graph.nodes['recipe'].data['instr_feature'] = recipe_instruction_features
    graph.nodes['ingredient'].data['nutrient_feature'] = ingredient_features
    graph.nodes['recipe'].data['train_mask'] = train_mask
    graph.nodes['recipe'].data['val_mask'] = val_mask
    graph.nodes['recipe'].data['test_mask'] = test_mask
    graph.nodes['recipe'].data['label'] = recipe_nodes_labels.long()

    return graph