import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from itertools import product
import matplotlib.patches as mpatches


fashion_data = pd.read_csv('./intermediate_files/closest_images_fashion.csv')
lfw_data = pd.read_csv('./intermediate_files/closest_images_lfw.csv')
combined_data = pd.read_csv('./intermediate_files/closest_images_overall.csv')
swatch_lightform = pd.read_csv('./intermediate_files/morphe_swatch_lightform.csv')
swatch_clustered = pd.read_csv('./intermediate_files/morphe_swatch_clustered.csv')

def split_by_underscore(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits each Morphe shade match by underscore and selects the first part of the split text, i.e. the broad skin-tone category in the Morphe Lightform quiz.
    Returns: 
    - morphe_df: DataFrame of Morphe predictions by broad category.
    - clustered_df: DataFrame of clustered predictions by broad category.
    """
    cols_clustered = [f'RJ{x}' for x in range(1, 6)]
    cols_morphe = [f'Morphe{x}' for x in range(1, 6)]

    # Process columns to extract text before underscore
    morphe_df = df[cols_morphe].applymap(lambda x: x.split('_')[0] if isinstance(x, str) else x)
    clustered_df = df[cols_clustered].applymap(lambda x: x.split('_')[0] if isinstance(x, str) else x)

    return morphe_df, clustered_df

def plot_results(morphe_df: pd.DataFrame, clustered_df: pd.DataFrame, title: str) -> None:
    """
    Create a side-by-side bar plot comparing Morphe and Clustered data.

    Args:
        morphe_df: DataFrame containing Morphe predictions.
        clustered_df: DataFrame containing Clustered predictions.
        title: Title of the plot.
    """
    # Aggregate data
    morphe_counts = morphe_df.apply(pd.Series.value_counts).fillna(0).sum(axis=1)
    clustered_counts = clustered_df.apply(pd.Series.value_counts).fillna(0).sum(axis=1)

    # Ensure both have the same categories
    all_categories = sorted(set(morphe_counts.index).union(set(clustered_counts.index)))
    morphe_counts = morphe_counts.reindex(all_categories, fill_value=0)
    clustered_counts = clustered_counts.reindex(all_categories, fill_value=0)

    # Create side-by-side bar plot
    x = range(len(all_categories))  # X-axis positions for categories
    bar_width = 0.4  # Width of each bar

    morphe_color_map = {
        'deep':(0.6392156863,0.3176470588,0.1411764706),
        'deepest':(0.4196078431,0.1725490196,0.05098039216),
        'deeprich':(0.8431372549,0.4549019608,0.2156862745),
        'deeptan':(0.8823529412,0.5411764706,0.3529411765),
        'fairlight':(0.9450980392,0.7607843137,0.6470588235),
        'light':(1,0.7490196078,0.6),
        'lightmedium':(1,0.7647058824,0.5960784314),
        'medium':(0.9960784314 ,   0.7058823529  , 0.5254901961),
        'rich':(0.8431372549,0.5176470588,0.3019607843),
        'tan': (0.968627451,0.6549019608,0.4588235294)
    }

    clusterd_color_map  = {
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
    clusters_patch = mpatches.Patch(color='springgreen', label='Clustered')
    plt.legend(handles=[lightform_patch, clusters_patch], title = "Dataset")

    plt.bar([pos - bar_width / 2 for pos in x], morphe_counts, bar_width, label='Morphe', color = [morphe_color_map[i] for i in morphe_counts.index],edgecolor= 'salmon', linewidth=3)
    plt.bar([pos + bar_width / 2 for pos in x], clustered_counts, bar_width, label='Clustered', color = [clusterd_color_map[i] for i in clustered_counts.index], edgecolor = 'springgreen', linewidth=3)

    # Add labels, legend, and title
    plt.xticks(x, all_categories, rotation=45)
    plt.xlabel("Categories")
    plt.ylabel("Count")
    plt.title(title)
  
    plt.tight_layout()
    plt.savefig(f'plots/{title}.png')

    plt.show()






morphe_df_fashion, clustered_df_fashion = split_by_underscore(fashion_data)
morphe_df_lfw, clustered_df_lfw = split_by_underscore(lfw_data)
morphe_df_combined, clustered_df_combined = split_by_underscore(combined_data)

print("Fashion Data Morphe:")
print(morphe_df_fashion.head())
print("Fashion Data Clustered:")
print(clustered_df_fashion.head())
print("LFW Data Morphe:")
print(morphe_df_lfw.head())
print("LFW Data Clustered:")
print(clustered_df_lfw.head())
print("Combined Data Morphe:")
print(morphe_df_combined.head())
print("Combined Data Clustered:")
print(clustered_df_combined.head())

#plot([morphe_df_fashion, morphe_df_lfw, morphe_df_combined], [clustered_df_fashion, clustered_df_lfw, clustered_df_combined])

plot_results(morphe_df_fashion, clustered_df_fashion, "Fashion Morphe vs Clustered")
plot_results(morphe_df_lfw, clustered_df_lfw, "LFW Morphe vs Clustered")
plot_results(morphe_df_combined, clustered_df_combined, "Combined Morphe vs Clustered")


    
