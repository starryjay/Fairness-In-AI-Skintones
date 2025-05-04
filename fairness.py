import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stats
import random

random.seed(39)

def get_agreement(gtpath: str, morphepath: str, clusterpath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt = pd.read_csv(gtpath).loc[:, ['Image Path', 'ground_truth1', 'ground_truth2', 'ground_truth3', 
                                     'ground_truth4', 'ground_truth5']]
    morphe = pd.read_csv(morphepath).loc[:, ['Image Path', 'Morphe1', 'Morphe2', 'Morphe3', 'Morphe4', 'Morphe5']]
    cluster = pd.read_csv(clusterpath).loc[:, ['Image Path', 'RJ1', 'RJ2', 'RJ3', 'RJ4', 'RJ5']]
    gt_morphe = gt.merge(morphe, on='Image Path', how='inner')
    skintone_map = {'fairlight': 'light', 'light': 'light', 'lightmedium': 'light',
                    'medium': 'medium', 'tan': 'medium', 'deeptan': 'medium', 
                    'rich': 'dark', 'deeprich': 'dark', 'deep': 'dark', 'deepest': 'dark'}
    gt_morphe.loc[:, 'Agreement'] = gt_morphe.apply(lambda x: len(set([x['ground_truth1'], x['ground_truth2'], x['ground_truth3'], x['ground_truth4'], x['ground_truth5']]).intersection(set([x['Morphe1'], 
                                                                       x['Morphe2'], x['Morphe3'], x['Morphe4'], 
                                                                       x['Morphe5']]))), axis=1)
    gt_morphe.loc[:, 'accurate'] = gt_morphe.apply(lambda x: x['Agreement'] >= 3, axis=1).astype(int)
    gt_morphe.loc[:, 'Skin Tone Category'] = gt_morphe.apply(lambda x: skintone_map[stats.mode([x['ground_truth1'].split('_')[0],
                                                                                       x['ground_truth2'].split('_')[0],
                                                                                       x['ground_truth3'].split('_')[0],
                                                                                       x['ground_truth4'].split('_')[0],
                                                                                       x['ground_truth5'].split('_')[0]])], axis=1)
    gt_cluster = gt.merge(cluster, on='Image Path', how='inner')
    gt_cluster.loc[:, 'Agreement'] = gt_cluster.apply(lambda x: len(set([x['ground_truth1'], x['ground_truth2'], 
                                                                         x['ground_truth3'], x['ground_truth4'], 
                                                                         x['ground_truth5']]).intersection(set([x['RJ1'], 
                                                                         x['RJ2'], x['RJ3'], x['RJ4'], x['RJ5']]))), axis=1)
    gt_cluster.loc[:, 'accurate'] = gt_cluster.apply(lambda x: x['Agreement'] >= 3, axis=1).astype(int)
    gt_cluster.loc[:, 'Skin Tone Category'] = gt_cluster.apply(lambda x: skintone_map[stats.mode([x['ground_truth1'].split('_')[0], 
                                                                                       x['ground_truth2'].split('_')[0],
                                                                                       x['ground_truth3'].split('_')[0],
                                                                                       x['ground_truth4'].split('_')[0],
                                                                                       x['ground_truth5'].split('_')[0]])], axis=1)
    grouped_morphe = gt_morphe.groupby('Skin Tone Category').agg({'accurate': 'mean'}).reset_index()
    grouped_cluster = gt_cluster.groupby('Skin Tone Category').agg({'accurate': 'mean'}).reset_index()
    return grouped_morphe, grouped_cluster
    
def plot_agreement(gt_morphe: pd.DataFrame, gt_cluster: pd.DataFrame) -> None:
    """
    Plots the agreement between ground truth and Morphe/cluster.
    """
    plt.figure(figsize=(10, 6), dpi=300)
    sns.histplot(gt_morphe['Agreement'], bins=5, kde=True, color='deepskyblue', edgecolor='black', linewidth=0.8)
    plt.title('Agreement between Ground Truth and Morphe')
    plt.xlabel('Number of Agreements')
    plt.ylabel('Frequency')
    plt.savefig('./plots/agreement_morphe.png')
    plt.show()
    
    plt.figure(figsize=(10, 6), dpi=300)
    sns.histplot(gt_cluster['Agreement'], bins=5, kde=True, color='deepskyblue', edgecolor='black', linewidth=0.8)
    plt.title('Agreement between Ground Truth and Cluster')
    plt.xlabel('Number of Agreements')
    plt.ylabel('Frequency')
    plt.savefig('./plots/agreement_cluster.png')
    plt.show()
    
def fairness(grouped_morphe: pd.DataFrame, grouped_cluster: pd.DataFrame) -> None:
    parity_light_to_medium_morphe = abs(grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'light', 'accurate'].values[0] -
                                   grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'medium', 'accurate'].values[0])
    parity_medium_to_dark_morphe = abs(grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'medium', 'accurate'].values[0] -
                                      grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'dark', 'accurate'].values[0])
    parity_light_to_dark_morphe = abs(grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'light', 'accurate'].values[0] -
                                    grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'dark', 'accurate'].values[0])
    print(f'Parity light to medium for Morphe: {parity_light_to_medium_morphe}')
    print(f'Parity medium to dark for Morphe: {parity_medium_to_dark_morphe}')
    print(f'Parity light to dark for Morphe: {parity_light_to_dark_morphe}')
    print('---------------------------------------------------------------')
    parity_light_to_medium_cluster = abs(grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'light', 'accurate'].values[0] -
                                      grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'medium', 'accurate'].values[0])
    parity_medium_to_dark_cluster = abs(grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'medium', 'accurate'].values[0] -
                                        grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'dark', 'accurate'].values[0])
    parity_light_to_dark_cluster = abs(grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'light', 'accurate'].values[0] -
                                        grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'dark', 'accurate'].values[0])
    print(f'Parity light to medium for our clustering model: {parity_light_to_medium_cluster}')
    print(f'Parity medium to dark for our clustering model: {parity_medium_to_dark_cluster}')
    print(f'Parity light to dark for our clustering model: {parity_light_to_dark_cluster}')
    print('---------------------------------------------------------------')
    disparate_impact_medium_over_light_morphe = grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'medium', 'accurate'].values[0] / \
                                              grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'light', 'accurate'].values[0] if \
                                               grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'light', 'accurate'].values[0] != 0 \
                                                else float('inf')
    disparate_impact_dark_over_medium_morphe = grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'dark', 'accurate'].values[0] / \
                                                grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'medium', 'accurate'].values[0] if \
                                                  grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'medium', 'accurate'].values[0] != 0 \
                                                    else float('inf')
    disparate_impact_dark_over_light_morphe = grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'dark', 'accurate'].values[0] / \
                                                grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'light', 'accurate'].values[0] if \
                                                 grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'light', 'accurate'].values[0] != 0 \
                                                    else float('inf')
    
    disparate_impact_medium_over_light_cluster = grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'medium', 'accurate'].values[0] / \
                                                grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'light', 'accurate'].values[0] if \
                                                 grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'light', 'accurate'].values[0] != 0 \
                                                    else float('inf')
    disparate_impact_dark_over_medium_cluster = grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'dark', 'accurate'].values[0] / \
                                                grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'medium', 'accurate'].values[0] if \
                                                 grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'medium', 'accurate'].values[0] != 0 \
                                                    else float('inf')
    disparate_impact_dark_over_light_cluster = grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'dark', 'accurate'].values[0] / \
                                                grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'light', 'accurate'].values[0] if \
                                                 grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'light', 'accurate'].values[0] != 0 \
                                                    else float('inf')
    
    disparate_impact_light_over_medium_morphe = grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'light', 'accurate'].values[0] / \
                                              grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'medium', 'accurate'].values[0] if \
                                               grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'medium', 'accurate'].values[0] != 0 \
                                                else float('inf')
    disparate_impact_medium_over_dark_morphe = grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'medium', 'accurate'].values[0] / \
                                                grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'dark', 'accurate'].values[0] if \
                                                 grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'dark', 'accurate'].values[0] != 0 \
                                                    else float('inf')
    disparate_impact_light_over_dark_morphe = grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'light', 'accurate'].values[0] / \
                                                grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'dark', 'accurate'].values[0] if \
                                                 grouped_morphe.loc[grouped_morphe['Skin Tone Category'] == 'dark', 'accurate'].values[0] != 0 \
                                                    else float('inf')
    
    disparate_impact_light_over_medium_cluster = grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'light', 'accurate'].values[0] / \
                                              grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'medium', 'accurate'].values[0] if \
                                               grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'medium', 'accurate'].values[0] != 0 \
                                                else float('inf')
    disparate_impact_medium_over_dark_cluster = grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'medium', 'accurate'].values[0] / \
                                                grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'dark', 'accurate'].values[0] if \
                                                 grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'dark', 'accurate'].values[0] != 0 \
                                                    else float('inf')
    disparate_impact_light_over_dark_cluster = grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'light', 'accurate'].values[0] / \
                                                grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'dark', 'accurate'].values[0] if \
                                                 grouped_cluster.loc[grouped_cluster['Skin Tone Category'] == 'dark', 'accurate'].values[0] != 0 \
                                                    else float('inf')

    print(f'Disparate impact medium over light for Morphe: {disparate_impact_medium_over_light_morphe}')
    print(f'Disparate impact dark over medium for Morphe: {disparate_impact_dark_over_medium_morphe}')
    print(f'Disparate impact dark over light for Morphe: {disparate_impact_dark_over_light_morphe}')
    print('---------------------------------------------------------------')
    print(f'Disparate impact medium over light for our clustering model: {disparate_impact_medium_over_light_cluster}')
    print(f'Disparate impact dark over medium for our clustering model: {disparate_impact_dark_over_medium_cluster}')
    print(f'Disparate impact dark over light for our clustering model: {disparate_impact_dark_over_light_cluster}')
    print('---------------------------------------------------------------')
    print(f'Disparate impact light over medium for Morphe: {disparate_impact_light_over_medium_morphe}')
    print(f'Disparate impact medium over dark for Morphe: {disparate_impact_medium_over_dark_morphe}')
    print(f'Disparate impact light over dark for Morphe: {disparate_impact_light_over_dark_morphe}')
    print('---------------------------------------------------------------')
    print(f'Disparate impact light over medium for our clustering model: {disparate_impact_light_over_medium_cluster}')
    print(f'Disparate impact medium over dark for our clustering model: {disparate_impact_medium_over_dark_cluster}')
    print(f'Disparate impact light over dark for our clustering model: {disparate_impact_light_over_dark_cluster}')
    print('---------------------------------------------------------------')
    
def main():
    gtpath = './intermediate_files/ground_truth2.csv'
    morphepath = './intermediate_files/closest_images_overall.csv'
    clusterpath = './intermediate_files/closest_images_overall_mod.csv'
    gt_morphe, gt_cluster = get_agreement(gtpath, morphepath, clusterpath)
    gt_morphe.to_csv('./intermediate_files/gt_morphe_agreement.csv', index=False)
    gt_cluster.to_csv('./intermediate_files/gt_cluster_agreement.csv', index=False)

    # print('Agreement between ground truth and Morphe:')
    # print(gt_morphe.head())
    # print('Agreement between ground truth and cluster:')
    # print(gt_cluster.head())

    print('Fairness metrics:')
    print('---------------------------------------------------------------')
    fairness(gt_morphe, gt_cluster)

    #plot_agreement(gt_morphe, gt_cluster)

if __name__ == "__main__":
    main()