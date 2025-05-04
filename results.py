import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def transform_data(gtpath: str, closestpath: str, closestmodpath: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ground_truth = pd.read_csv(gtpath)
    ground_truth = ground_truth.loc[:, ['Image Path', 'ground_truth1', 'ground_truth2', 'ground_truth3', 'ground_truth4', 'ground_truth5']]
    ground_truth = ground_truth.rename(columns={'Image Path': 'Image'})
    closest = pd.read_csv(closestpath)
    closestmod = pd.read_csv(closestmodpath)
    closest_morphe = closest.loc[:, ['Image Path', 'Morphe1', 'Morphe2', 'Morphe3', 'Morphe4', 'Morphe5']]
    closest_morphe = closest_morphe.rename(columns={'Image Path': 'Image'})
    closest_cluster = closestmod.loc[:, ['Image Path', 'RJ1', 'RJ2', 'RJ3', 'RJ4', 'RJ5']]
    closest_cluster = closest_cluster.rename(columns={'Image Path': 'Image'})
    return ground_truth, closest_morphe, closest_cluster

def results_table(ground_truth, closest_morphe, closest_cluster):
    merged_morphe = pd.merge(ground_truth, closest_morphe, on='Image')
    merged = pd.merge(merged_morphe, closest_cluster, on='Image')
    print(merged.columns)

    merged['Jaccard_Morphe_Exact'] = merged.apply(lambda row: len(set(
        row[['ground_truth1', 'ground_truth2', 'ground_truth3', 'ground_truth4', 'ground_truth5']]
        ) & \
            set(
                row[['Morphe1', 'Morphe2', 'Morphe3', 'Morphe4', 'Morphe5']]
                )) / len(set(
                row[['ground_truth1', 'ground_truth2', 'ground_truth3', 'ground_truth4', 'ground_truth5']]
                ) | \
                    set(
                        row[['Morphe1', 'Morphe2', 'Morphe3', 'Morphe4', 'Morphe5']]
                        )), axis=1)
    
    merged['Jaccard_Cluster_Exact'] = merged.apply(lambda row: len(set(
        row[['ground_truth1', 'ground_truth2', 'ground_truth3', 'ground_truth4', 'ground_truth5']]
        ) & \
            set(
                row[['RJ1', 'RJ2', 'RJ3', 'RJ4', 'RJ5']]
                )) / len(set(
                row[['ground_truth1', 'ground_truth2', 'ground_truth3', 'ground_truth4', 'ground_truth5']]
                ) | \
                    set(
                        row[['RJ1', 'RJ2', 'RJ3', 'RJ4', 'RJ5']]
                        )), axis=1)
    
    merged['Jaccard_Morphe_Broad'] = merged.apply(lambda row: len(set(
        [x.split('_')[0] if type(x) == str else x for x in row[['ground_truth1', 'ground_truth2', 'ground_truth3', 'ground_truth4', 'ground_truth5']]]
        ) & \
            set(
                [x.split('_')[0] if type(x) == str else x for x in row[['Morphe1', 'Morphe2', 'Morphe3', 'Morphe4', 'Morphe5']]]
                )) / len(set(
                [x.split('_')[0] if type(x) == str else x for x in row[['ground_truth1', 'ground_truth2', 'ground_truth3', 'ground_truth4', 'ground_truth5']]]
                ) | \
                    set(
                        [x.split('_')[0] if type(x) == str else x for x in row[['Morphe1', 'Morphe2', 'Morphe3', 'Morphe4', 'Morphe5']]]
                        )), axis=1)

    merged['Jaccard_Cluster_Broad'] = merged.apply(lambda row: len(set(
        [x.split('_')[0] if type(x) == str else x for x in row[['ground_truth1', 'ground_truth2', 'ground_truth3', 'ground_truth4', 'ground_truth5']]]
        ) & \
            set(
                [x.split('_')[0] if type(x) == str else x for x in row[['RJ1', 'RJ2', 'RJ3', 'RJ4', 'RJ5']]]
                )) / len(set(
                [x.split('_')[0] if type(x) == str else x for x in row[['ground_truth1', 'ground_truth2', 'ground_truth3', 'ground_truth4', 'ground_truth5']]]
                ) | \
                    set(
                        [x.split('_')[0] if type(x) == str else x for x in row[['RJ1', 'RJ2', 'RJ3', 'RJ4', 'RJ5']]]
                        )), axis=1)

    merged['Raw_Matches_Morphe'] = merged.apply(lambda row: len(set(
        row[['ground_truth1', 'ground_truth2', 'ground_truth3', 'ground_truth4', 'ground_truth5']]
        ) & \
            set(
                row[['Morphe1', 'Morphe2', 'Morphe3', 'Morphe4', 'Morphe5']]
                )), axis=1)
    merged['Raw_Matches_Cluster'] = merged.apply(lambda row: len(set(
        row[['ground_truth1', 'ground_truth2', 'ground_truth3', 'ground_truth4', 'ground_truth5']]
        ) & \
            set(
                row[['RJ1', 'RJ2', 'RJ3', 'RJ4', 'RJ5']]
                )), axis=1)
    
    morphe_df = merged[['Morphe1', 'Morphe2', 'Morphe3', 'Morphe4', 'Morphe5']]
    clustered_df = merged[['RJ1', 'RJ2', 'RJ3', 'RJ4', 'RJ5']]
    ground_truth_df = merged[['ground_truth1', 'ground_truth2', 'ground_truth3', 'ground_truth4', 'ground_truth5']]
    morphe_df = morphe_df.map(lambda x: x.split('_')[0] if isinstance(x, str) else x)
    clustered_df = clustered_df.map(lambda x: x.split('_')[0] if isinstance(x, str) else x)
    ground_truth_df = ground_truth_df.map(lambda x: x.split('_')[0] if isinstance(x, str) else x)
    morphe_df.dropna(how='all', inplace=True)
    clustered_df.dropna(how='all', inplace=True)
    morphe_counts = morphe_df.apply(pd.Series.value_counts).fillna(0).sum(axis=1)
    clustered_counts = clustered_df.apply(pd.Series.value_counts).fillna(0).sum(axis=1)
    ground_truth_counts = ground_truth_df.apply(pd.Series.value_counts).fillna(0).sum(axis=1)
    all_categories = ['fairlight', 'light', 'lightmedium', 'medium', 'tan', 'deeptan', 'rich', 'deeprich', 'deep', 'deepest']
    morphe_counts = morphe_counts.reindex(all_categories, fill_value=0)
    clustered_counts = clustered_counts.reindex(all_categories, fill_value=0)
    ground_truth_counts = ground_truth_counts.reindex(all_categories, fill_value=0)

    counts_df = pd.DataFrame({
        'Morphe': morphe_counts,
        'Clustered': clustered_counts,
        'Ground Truth': ground_truth_counts
    })

    counts_df['Morphe_Deviation'] = (counts_df['Morphe'] - counts_df['Ground Truth']) / counts_df['Ground Truth'] * 100
    counts_df['Clustered_Deviation'] = (counts_df['Clustered'] - counts_df['Ground Truth']) / counts_df['Ground Truth'] * 100
    print(counts_df)

    return merged, counts_df

def plot_results(merged: pd.DataFrame, counts_df: pd.DataFrame) -> None:
    morphe_df = merged[['Morphe1', 'Morphe2', 'Morphe3', 'Morphe4', 'Morphe5']]
    clustered_df = merged[['RJ1', 'RJ2', 'RJ3', 'RJ4', 'RJ5']]
    ground_truth_df = merged[['ground_truth1', 'ground_truth2', 'ground_truth3', 'ground_truth4', 'ground_truth5']]
    morphe_df = morphe_df.applymap(lambda x: x.split('_')[0] if isinstance(x, str) else x)
    clustered_df = clustered_df.applymap(lambda x: x.split('_')[0] if isinstance(x, str) else x)
    ground_truth_df = ground_truth_df.applymap(lambda x: x.split('_')[0] if isinstance(x, str) else x)
    morphe_df.dropna(how='all', inplace=True)
    clustered_df.dropna(how='all', inplace=True)
    morphe_counts = morphe_df.apply(pd.Series.value_counts).fillna(0).sum(axis=1)
    clustered_counts = clustered_df.apply(pd.Series.value_counts).fillna(0).sum(axis=1)
    ground_truth_counts = ground_truth_df.apply(pd.Series.value_counts).fillna(0).sum(axis=1)

    all_categories = ['fairlight', 'light', 'lightmedium', 'medium', 'tan', 'deeptan', 'rich', 'deeprich', 'deep', 'deepest']
    morphe_counts = morphe_counts.reindex(all_categories, fill_value=0)

    clustered_counts = clustered_counts.reindex(all_categories, fill_value=0)
    ground_truth_counts = ground_truth_counts.reindex(all_categories, fill_value=0)

    x = range(len(all_categories))
    bar_width = 0.25

    morphe_color_map = {
        'deep':(0.6392156863, 0.3176470588, 0.1411764706),
        'deepest':(0.4196078431, 0.1725490196, 0.05098039216),
        'deeprich':(0.8431372549, 0.4549019608, 0.2156862745),
        'deeptan':(0.8823529412, 0.5411764706, 0.3529411765),
        'fairlight':(0.9450980392, 0.7607843137, 0.6470588235),
        'light':(1, 0.7490196078, 0.6),
        'lightmedium':(1, 0.7647058824, 0.5960784314),
        'medium':(0.9960784314, 0.7058823529, 0.5254901961),
        'rich':(0.8431372549, 0.5176470588, 0.3019607843),
        'tan': (0.968627451, 0.6549019608, 0.4588235294)
    }
    

    clustered_color_map  = {
        'deep': (0.5450980392,0.2941176471 , 0.1568627451),
        'deepest': (0.4980392157,0.2235294118,0.1333333333),
        'deeprich' : (0.6274509804,0.2941176471,0.03529411765),
        'deeptan': (0.9647058824,0.5568627451,0.3215686275),
        'fairlight': (1,0.7960784314,0.6392156863),
        'light': (1,0.7450980392,0.5647058824),
        'medium': (0.9882352941,0.7019607843,0.4941176471),
        'lightmedium': (0.9921568627 ,0.7215686275 ,0.5294117647),
        'rich': (0.8784313725,0.4862745098,0.2),
        'tan': (0.9803921569,0.6235294118,0.4)

    }
  
    lightform_patch = mpatches.Patch(color='salmon', label='Morphe')
    clusters_patch = mpatches.Patch(color='limegreen', label='Clustered')
    ground_truth_patch = mpatches.Patch(color='blue', label='Ground Truth')
    plt.legend(handles=[lightform_patch, clusters_patch, ground_truth_patch], title = "Dataset")
    plt.bar([pos - bar_width for pos in x], morphe_counts, bar_width, label='Morphe', color = [morphe_color_map[i] for i in morphe_counts.index],edgecolor= 'salmon', linewidth=2)
    plt.bar([pos for pos in x], ground_truth_counts, bar_width, label='Ground Truth', color = [morphe_color_map[i] for i in ground_truth_counts.index], edgecolor = 'blue', linewidth=2)
    plt.bar([pos + bar_width for pos in x], clustered_counts, bar_width, label='Clustered', color = [clustered_color_map[i] for i in clustered_counts.index], edgecolor = 'limegreen', linewidth=2)
    #plt.fill_between(x, morphe_counts, color='salmon', alpha=0.3)
    #plt.fill_between(x, clustered_counts, color='limegreen', alpha=0.3)
    #plt.fill_between(x, ground_truth_counts, color='blue', alpha=0.3)

    # Add labels, legend, and title
    plt.xticks(x, all_categories, rotation=45)
    plt.xlabel("Categories")
    plt.ylabel("Count")
    plt.title("Skin Tone Distribution Comparison")
  
    plt.tight_layout()
    plt.savefig(f'plots/skin_tone_distribution_gt.png')

    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(counts_df.index, counts_df['Morphe_Deviation'], color='salmon', label='Morphe Deviation')
    plt.bar(counts_df.index, counts_df['Clustered_Deviation'], color='limegreen', label='Clustered Deviation')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xticks(rotation=45)
    plt.xlabel("Categories")
    plt.ylabel("Percentage Deviation from Ground Truth")
    plt.title("Percentage Deviation from Ground Truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/percentage_deviation.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    counts_df = counts_df.drop(['deeptan', 'rich', 'deep', 'deepest'])
    plt.bar(counts_df.index, counts_df['Morphe_Deviation'], color='salmon', label='Morphe Deviation')
    plt.bar(counts_df.index, counts_df['Clustered_Deviation'], color='limegreen', label='Clustered Deviation')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xticks(rotation=45)    
    
def main():
    gtpath = './intermediate_files/ground_truth.csv'
    closestpath = './intermediate_files/closest_images_overall.csv'
    closestmodpath = './intermediate_files/closest_images_overall_mod.csv'

    ground_truth, closest_morphe, closest_cluster = transform_data(gtpath, closestpath, closestmodpath)
    results, counts_df = results_table(ground_truth, closest_morphe, closest_cluster)
    plot_results(results, counts_df)

    results.to_csv('./intermediate_files/similarity.csv', index=False)

if __name__ == "__main__":
    main()
