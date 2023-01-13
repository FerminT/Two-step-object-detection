import json
import pandas as pd

def replaceSubcat(subcatlst, labels):
    for lab in subcatlst:
        if 'LabelName' in lab:
            labelstr = lab['LabelName']
            dfrow = labels.loc[labels['LabelName'] == labelstr]
            if not dfrow.empty:
                lab['LabelName'] = dfrow['DisplayName'].iloc[0]
        if 'Subcategory' in lab:
            replaceSubcat(lab['Subcategory'], labels)
        if 'Part' in lab:
            replaceSubcat(lab['Part'], labels)

with open('bbox_labels_600_hierarchy.json', 'r') as fp:
    hierarchy = json.load(fp)

labels = pd.read_csv('class-descriptions-boxable.csv')
replaceSubcat(hierarchy['Subcategory'], labels)

with open('hierarchy.json', 'w') as fp:
    json.dump(hierarchy, fp, indent=4)    