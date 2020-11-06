import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")

PROJECT_NAME = sys.environ('PROJECT_NAME')
COMP_NAME = sys.environ('COMP_NAME')
MODEL_NAME = 'my_model.pth'

sys.path.insert(1, f'/kaggle/input/{PROJECT_NAME}-code')
sys.path.insert(1, '../code')

from inference import ProjectInferrer

if __name__ == '__main__':

    images_path = f'/kaggle/input/{COMP_NAME}/test/'
    model_path = f'/kaggle/input/{PROJECT_NAME}-models/{MODEL_NAME}'
    submission_path = f'/kaggle/input/{COMP_NAME}/sample_submission.csv'

    submission = pd.read_csv(submission_path)

    inferrer = ProjectInferrer(images_path, model_path)
    for i, case in enumerate(submission['image_name']):
        print(i+1, case, '...')
        pred = inferrer(case)
        submission['target'][submission['image_name'] == case] = pred
        print(pred)

    print('saving output...')
    submission.to_csv('submission.csv', index=False)
    print('done!')
