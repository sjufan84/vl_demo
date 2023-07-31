import pandas as pd

def prepare_data_for_plot(tsne_results):
    data = []
    for file_name, result in tsne_results.items():
        for point in result:
            data.append((file_name, point[0], point[1]))

    df = pd.DataFrame(data, columns=["File Name", "Dimension 1", "Dimension 2"])
    return df
