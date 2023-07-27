model_dict = {
    'aro': {'GRU': {'biosig': ['RNN_2023-06-13-12-22_[biosignals]_[physio-arousal]_[32_8_False_32]_[0.005_256]_400',
                               '117_personalised_2023-07-03-02-53-15'],
                    'data2vec audio': [
                        'RNN_2023-06-17-05-04_[data2vec_audio]_[physio-arousal]_[16_2_False_16]_[0.005_256]_400',
                        '110_personalised_2023-07-03-05-33-40'],
                    'data2vec context': [
                        'RNN_2023-06-20-12-12_[data2vec_context]_[physio-arousal]_[64_2_False_32]_[0.005_256]_400',
                        '105_personalised_2023-07-03-05-32-07'],
                    'deepspectrum': ['RNN_2023-06-09-11-03_[ds]_[physio-arousal]_[64_2_False_64]_[0.005_256]_200',
                                     '108_personalised_2023-07-03-02-57-51'],
                    'egemaps': ['RNN_2023-06-08-01-47_[egemaps]_[physio-arousal]_[32_8_False_64]_[0.001_256]_400',
                                '120_personalised_2023-07-03-02-58-21'],
                    'facenet': ['RNN_2023-06-14-02-00_[facenet]_[physio-arousal]_[32_8_False_64]_[0.002_256]_200',
                                '120_personalised_2023-07-03-11-56-19'],
                    'fau': ['RNN_2023-06-12-15-10_[faus]_[physio-arousal]_[32_2_False_32]_[0.002_256]_400',
                            '105_personalised_2023-07-03-11-57-02'],
                    'pose1': ['RNN_2023-06-16-14-37_[pose]_[physio-arousal]_[128_4_False_64]_[0.005_256]_300',
                              '120_personalised_2023-07-03-05-30-46'],
                    'pose_c': ['RNN_2023-06-24-04-16_[pose_c]_[physio-arousal]_[64_4_False_16]_[0.005_256]_400',
                               '106_personalised_2023-07-03-05-31-52'],
                    'pose_r': ['RNN_2023-06-23-11-55_[pose_r]_[physio-arousal]_[64_4_False_64]_[0.01_256]_400',
                               '108_personalised_2023-07-03-14-33-13'],
                    'pose_rc': ['RNN_2023-06-23-09-08_[pose_rc]_[physio-arousal]_[64_4_False_32]_[0.005_256]_200',
                                '107_personalised_2023-07-03-14-32-49'],
                    'vit': ['RNN_2023-06-15-04-39_[vit]_[physio-arousal]_[32_4_False_64]_[0.005_256]_400',
                            '107_personalised_2023-07-03-02-55-43'],
                    'wav2vec audio': [
                        'RNN_2023-06-18-06-41_[wav2vec_audio]_[physio-arousal]_[16_4_False_32]_[0.005_256]_400',
                        '112_personalised_2023-07-03-05-34-00'],
                    'wav2vec context': [
                        'RNN_2023-06-18-18-55_[wav2vec_context]_[physio-arousal]_[16_2_False_16]_[0.005_256]_400',
                        '112_personalised_2023-07-03-15-11-17'],
                    'wave2vec': ['RNN_2023-06-21-09-52_[w2v-msp]_[physio-arousal]_[64_2_False_16]_[0.005_256]_400',
                                 '105_personalised_2023-07-03-04-06-22']},
            'TF': {'biosig': ['TF_2023-06-13-07-01_[biosignals]_[physio-arousal]_[128_2_False_64]_[0.005_256]_200',
                              '112_personalised_2023-07-03-12-28-35'],
                   'data2vec audio': [
                       'TF_2023-06-17-19-37_[data2vec_audio]_[physio-arousal]_[64_2_False_16]_[0.005_256]_400',
                       '114_personalised_2023-07-03-06-10-06'],
                   'data2vec context': [
                       'TF_2023-06-20-06-46_[data2vec_context]_[physio-arousal]_[64_2_False_64]_[0.005_256]_400',
                       '110_personalised_2023-07-03-15-17-21'],
                   'deepspectrum': ['TF_2023-06-09-23-32_[ds]_[physio-arousal]_[32_2_False_64]_[0.005_256]_200',
                                    '119_personalised_2023-07-03-02-59-37'],
                   'egemaps': ['TF_2023-06-30-14-41_[egemaps]_[physio-arousal]_[64_4_False_64]_[0.002_256]_400',
                               '117_personalised_2023-07-05-21-14-30'],
                   'facenet': ['TF_2023-06-14-20-31_[facenet]_[physio-arousal]_[64_4_False_32]_[0.005_256]_400',
                               '103_personalised_2023-07-03-03-31-39'],
                   'fau': ['TF_2023-06-11-17-33_[faus]_[physio-arousal]_[32_2_False_64]_[0.005_256]_400',
                           '107_personalised_2023-07-03-03-34-47'],
                   'pose1': ['TF_2023-06-17-04-50_[pose]_[physio-arousal]_[32_4_False_64]_[0.002_256]_200',
                             '117_personalised_2023-07-03-15-11-37'],
                   'pose_c': ['TF_2023-06-23-17-00_[pose_c]_[physio-arousal]_[64_4_False_16]_[0.01_256]_400',
                              '114_personalised_2023-07-03-05-36-43'],
                   'pose_r': ['TF_2023-06-23-07-54_[pose_r]_[physio-arousal]_[64_8_False_16]_[0.005_256]_400',
                              '114_personalised_2023-07-03-06-00-46'],
                   'pose_rc': ['TF_2023-06-22-02-04_[pose_rc]_[physio-arousal]_[16_8_False_16]_[0.005_256]_200',
                               '107_personalised_2023-07-03-15-58-11'],
                   'vit': ['TF_2023-06-15-08-49_[vit]_[physio-arousal]_[32_2_False_32]_[0.005_256]_200',
                           '117_personalised_2023-07-05-12-14-00'],
                   'wav2vec audio': [
                       'TF_2023-06-18-18-32_[wav2vec_audio]_[physio-arousal]_[64_2_False_64]_[0.005_256]_400',
                       '101_personalised_2023-07-03-06-01-07'],
                   'wav2vec context': [
                       'TF_2023-06-19-01-44_[wav2vec_context]_[physio-arousal]_[64_2_False_32]_[0.005_256]_400',
                       '102_personalised_2023-07-03-05-56-25'],
                   'wave2vec': ['TF_2023-06-21-14-02_[w2v-msp]_[physio-arousal]_[16_2_False_64]_[0.005_256]_400',
                                '111_personalised_2023-07-03-12-01-42']}},
    'val': {'GRU': {'biosig': ['RNN_2023-06-13-07-28_[biosignals]_[valence]_[64_4_False_32]_[0.005_256]_400',
                               '111_personalised_2023-07-03-04-08-21'],
                    'data2vec audio': [
                        'RNN_2023-06-18-06-30_[data2vec_audio]_[valence]_[64_4_False_64]_[0.002_256]_400',
                        '116_personalised_2023-07-03-06-20-36'],
                    'data2vec context': ['ignore', 'me'],
                    'deepspectrum': ['RNN_2023-06-10-03-57_[ds]_[valence]_[32_2_False_64]_[0.001_256]_200',
                                     '106_personalised_2023-07-03-04-47-26'],
                    'egemaps': ['RNN_2023-06-09-05-46_[egemaps]_[valence]_[128_4_False_64]_[0.001_256]_400',
                                '104_personalised_2023-07-03-04-08-51'],
                    'facenet': ['RNN_2023-06-14-19-16_[facenet]_[valence]_[32_8_False_32]_[0.002_256]_400',
                                '117_personalised_2023-07-03-12-02-10'],
                    'fau': ['RNN_2023-06-12-03-34_[faus]_[valence]_[64_8_False_32]_[0.005_256]_400',
                            '113_personalised_2023-07-03-12-02-32'],
                    'pose1': ['RNN_2023-06-17-00-44_[pose]_[valence]_[128_4_False_64]_[0.005_256]_400',
                              '120_personalised_2023-07-03-15-28-51'],
                    'pose_c': ['RNN_2023-06-29-12-04_[pose_c]_[valence]_[64_4_False_16]_[0.01_256]_400',
                               '120_personalised_2023-07-03-15-05-02'],
                    'pose_r': ['RNN_2023-06-23-20-46_[pose_r]_[valence]_[64_4_False_64]_[0.005_256]_400',
                               '118_personalised_2023-07-03-16-27-59'],
                    'pose_rc': ['RNN_2023-06-22-21-45_[pose_rc]_[valence]_[64_4_False_16]_[0.005_256]_400',
                                '106_personalised_2023-07-03-15-04-43'],
                    'vit': ['RNN_2023-06-16-06-32_[vit]_[valence]_[32_4_False_32]_[0.002_256]_400',
                            '114_personalised_2023-07-03-13-58-51'],
                    'wav2vec audio': ['RNN_2023-06-18-10-41_[wav2vec_audio]_[valence]_[32_2_False_64]_[0.002_256]_400',
                                      '113_personalised_2023-07-03-06-13-50'],
                    'wav2vec context': [
                        'RNN_2023-06-19-11-40_[wav2vec_context]_[valence]_[64_8_False_16]_[0.005_256]_400',
                        '101_personalised_2023-07-03-07-37-43'],
                    'wave2vec': ['RNN_2023-06-21-10-21_[w2v-msp]_[valence]_[64_4_False_32]_[0.005_256]_400',
                                 '120_personalised_2023-07-03-12-02-53']},
            'TF': {'biosig': ['TF_2023-06-13-00-59_[biosignals]_[valence]_[32_4_False_32]_[0.002_256]_200',
                              '120_personalised_2023-07-03-04-47-57'],
                   'data2vec audio': ['TF_2023-06-17-22-34_[data2vec_audio]_[valence]_[32_2_False_64]_[0.005_256]_400',
                                      '103_personalised_2023-07-03-06-39-09'],
                   'data2vec context': [
                       'TF_2023-06-20-14-36_[data2vec_context]_[valence]_[16_2_False_64]_[0.005_256]_400',
                       '109_personalised_2023-07-03-15-17-01'],
                   'deepspectrum': ['TF_2023-06-09-20-26_[ds]_[valence]_[32_2_False_32]_[0.002_256]_400',
                                    '111_personalised_2023-07-03-04-28-01'],
                   'egemaps': ['TF_2023-06-08-12-59_[egemaps]_[valence]_[64_4_False_64]_[0.002_256]_200',
                               '114_personalised_2023-07-03-05-27-55'],
                   'facenet': ['TF_2023-06-14-13-20_[facenet]_[valence]_[32_4_False_64]_[0.005_256]_400',
                               '114_personalised_2023-07-03-12-23-33'],
                   'fau': ['TF_2023-06-11-11-51_[faus]_[valence]_[32_2_False_64]_[0.005_256]_200',
                           '120_personalised_2023-07-03-12-23-57'],
                   'pose1': ['TF_2023-06-17-06-41_[pose]_[valence]_[64_4_False_64]_[0.002_256]_400',
                             '102_personalised_2023-07-03-06-22-56'],
                   'pose_c': ['TF_2023-06-23-17-02_[pose_c]_[valence]_[32_4_False_64]_[0.005_256]_400',
                              '107_personalised_2023-07-03-15-57-35'],
                   'pose_r': ['TF_2023-06-23-08-02_[pose_r]_[valence]_[32_8_False_16]_[0.01_256]_400',
                              '107_personalised_2023-07-03-06-52-28'],
                   'pose_rc': ['TF_2023-06-23-07-12_[pose_rc]_[valence]_[16_2_False_64]_[0.005_256]_400',
                               '104_personalised_2023-07-03-06-34-47'],
                   'vit': ['TF_2023-06-16-09-34_[vit]_[valence]_[128_2_False_64]_[0.002_256]_400',
                           '116_personalised_2023-07-03-03-44-22'],
                   'wav2vec audio': ['TF_2023-06-19-01-08_[wav2vec_audio]_[valence]_[16_4_False_64]_[0.002_256]_400',
                                     '109_personalised_2023-07-03-06-41-17'],
                   'wav2vec context': [
                       'TF_2023-06-18-21-19_[wav2vec_context]_[valence]_[16_4_False_16]_[0.002_256]_400',
                       '109_personalised_2023-07-03-15-29-12'],
                   'wave2vec': ['TF_2023-06-21-14-34_[w2v-msp]_[valence]_[64_2_False_64]_[0.005_256]_400',
                                '104_personalised_2023-07-03-12-47-00']}}}


def fusion_gen(emo_dim, models: list, features: list, weights=[]):
    #generates late fusion command
    model_ids = ""
    personalised = ""
    for idx, feature in enumerate(features):
        model = models[idx]
        model_id, personal_id = model_dict[emo_dim][model][feature]
        model_ids += model_id
        model_ids += " "
        personalised += personal_id
        personalised += " "
    model_ids = model_ids.strip()
    personalised = personalised.strip()
    if emo_dim == 'val':
        full_emo_dim = 'valence'
    elif emo_dim == 'aro':
        full_emo_dim = 'physio-arousal'
    command = f"python late_fusion.py --task personalisation --emo_dim {full_emo_dim} --model_ids {model_ids} --personalised {personalised}"
    if weights:
        weights = [str(x) for x in weights]
        command += f" --weights {' '.join(weights)}"
    print(command)
    return command


fusion_gen('aro', ['GRU', 'GRU', 'GRU', 'TF', 'GRU', 'TF'],
           ['vit', 'facenet', 'fau', 'deepspectrum', 'egemaps', 'wav2vec context'], weights=[1, 1, 2, 2, 1, 2])
fusion_gen('val', ['TF', 'TF', 'TF', 'TF', 'TF'], ['vit', 'facenet', 'deepspectrum', 'egemaps', 'wav2vec context'],
           weights=[1, 1, 1, 2, 2])
