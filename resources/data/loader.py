# Copyright 2019 Lukas Jendele and Ondrej Skopek.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import csv
import os

from dotmap import DotMap
import pandas as pd
import random


def init(breast_prefix):
    global bcdr_path, inbreast_path, cbis_path, zrh_path
    bcdr_path = os.path.join(breast_prefix, '1_BCDR')
    inbreast_path = os.path.join(breast_prefix, '2_INbreast')
    cbis_path = os.path.join(breast_prefix, 'cbis')
    zrh_path = os.path.join(breast_prefix, '3_zrh')


def load_all_datasets(dataset):

    def filter_view(images):
        images = filter_noisy_images(images)
        res_healthy = []
        res_cancer = []
        for image in images:
            # if image.view != 'CC':
            #     continue
            if image.cancer:
                res_cancer.append(image)
            else:
                res_healthy.append(image)
        return res_healthy, res_cancer

    def filter_noisy_images(images, filter_csv='data_in/Exclusions_BreastGAN.csv'):
        exclusions = pd.read_csv(filter_csv, header=0, delimiter=';')
        excluded = set(exclusions['Filename'])
        for _, image in images.items():
            basename = os.path.basename(image.image_path)
            if basename in excluded:
                print('Excluding file (Anton):', image.image_path)
            else:
                yield image

    def read_inbreast():
        print("Inbreast")
        images_inb, _ = load_inbreast()
        inb_healthy, inb_cancer = filter_view(images_inb)
        print("Healthy:", len(inb_healthy), "Cancer:", len(inb_cancer))
        print()
        return inb_healthy, inb_cancer

    def read_bcdr01():
        print("BCDR-D01")
        images_d01, _ = load_bcdr('BCDR-D01')
        d01_healthy, d01_cancer = filter_view(images_d01)
        print("Healthy:", len(d01_healthy), "Cancer:", len(d01_cancer))
        print()
        return d01_healthy, d01_cancer

    def read_bcdr02():
        print("BCDR-D02")
        images_d02, _ = load_bcdr('BCDR-D02')
        d02_healthy, d02_cancer = filter_view(images_d02)
        print("Healthy:", len(d02_healthy), "Cancer:", len(d02_cancer))
        print()
        return d02_healthy, d02_cancer

    def read_cbis():
        print("CBIS")
        images_cbis = read_cbis_csv(cbis_path)
        cbis_healthy, cbis_cancer = filter_view(images_cbis)
        print("Healthy:", len(cbis_healthy), "Cancer:", len(cbis_cancer))
        print()
        return cbis_healthy, cbis_cancer

    def read_zrh():
        print("ZRH")
        images_zrh = read_zrh_folder(zrh_path)
        zrh_healthy, zrh_cancer = filter_view(images_zrh)
        print("Healthy:", len(zrh_healthy), "Cancer:", len(zrh_cancer))
        print()
        return zrh_healthy, zrh_cancer

    def read_all():
        inb_healthy, inb_cancer = read_inbreast()
        d01_healthy, d01_cancer = read_bcdr01()
        d02_healthy, d02_cancer = read_bcdr02()
        cbis_healthy, cbis_cancer = read_cbis()
        zrh_healthy, zrh_cancer = read_zrh()
        print("Overall")
        healthy = inb_healthy + d01_healthy + d02_healthy + cbis_healthy + zrh_healthy
        cancer = inb_cancer + d01_cancer + d02_cancer + cbis_cancer + zrh_cancer
        print("Healthy:", len(healthy), "Cancer:", len(cancer))
        print()
        return healthy, cancer

    if dataset is None:
        healthy, cancer = read_all()
    elif dataset == 'inbreast':
        healthy, cancer = read_inbreast()
    elif dataset == 'bcdr01':
        healthy, cancer = read_bcdr01()
    elif dataset == 'bcdr02':
        healthy, cancer = read_bcdr02()
    elif dataset == 'cbis':
        healthy, cancer = read_cbis()
    elif dataset == 'zrh':
        healthy, cancer = read_zrh()
    else:
        raise ValueError('Unknown dataset ' + str(dataset))
    random.seed(42)
    random.shuffle(healthy)
    random.shuffle(cancer)
    return healthy, cancer


def get_cbis_image_path(folder, filename):
    split = filename.split('/')
    dir1_suffix = split[1][-5:]
    dir2_suffix = split[2][-5:]
    # print(dir1_suffix, dir2_suffix)
    filename = os.path.join(folder, 'CBIS-DDSM', split[0])
    dir1 = None
    for f in os.listdir(filename):
        if f.endswith(dir1_suffix):
            dir1 = f
            break
    if dir1 is None:
        raise ValueError('fail: ' + str(split))
    filename = os.path.join(filename, dir1)
    dir2 = None
    for f in os.listdir(filename):
        if f.endswith(dir2_suffix):
            dir2 = f
            break
    if dir2 is None:
        raise ValueError('fail: ' + str(split))
    filename = os.path.join(filename, dir2, split[-1]).strip()
    return filename


def read_cbis_csv(folder):
    files = ['mass_train.csv', 'mass_test.csv', 'calc_train.csv', 'calc_test.csv']
    df = pd.DataFrame()
    for file in files:
        file = os.path.join(folder, file)
        df = df.append(pd.read_csv(file), sort=True)

    df = df[df['image view'] == 'CC']
    lines = {}
    for i in range(0, len(df)):
        parsed = DotMap()
        parsed.dataset = 'cbis'
        parsed.cancer = df.iloc[i]['pathology'] == 'MALIGNANT'
        parsed.laterality = df.iloc[i]['left or right breast'][0]  # first char
        parsed.view = df.iloc[i]['image view']
        parsed.mask_path = [
                get_cbis_image_path(folder, df.iloc[i]['ROI mask file path']),
                get_cbis_image_path(folder, df.iloc[i]['cropped image file path'])
        ]
        parsed.image_path = get_cbis_image_path(folder, df.iloc[i]['image file path'])
        parsed.fid = i
        if parsed.fid in lines:
            raise ValueError('fid already exists: ' + str(parsed.fid))
        lines[parsed.fid] = parsed
    return lines


def read_zrh_folder(folder):

    def read_zrh_files(lines, folder, cancer, view):
        for file in os.listdir(folder):
            if not file.endswith('.png'):
                continue
            parsed = DotMap()
            parsed.dataset = 'zrh'
            parsed.cancer = cancer
            parsed.laterality = 'R'  # will get replaced anyway by flipping script
            parsed.view = view
            parsed.mask_path = None
            parsed.image_path = os.path.join(folder, file)
            parsed.fid = "zrh_{}_{}".format('cancer' if cancer else 'healthy', file.split('.')[0])
            if parsed.fid in lines:
                raise ValueError('fid already exists: ' + str(parsed.fid))
            lines[parsed.fid] = parsed

    cancer_folder = os.path.join(folder, 'cancer')
    healthy_folder = os.path.join(folder, 'control')
    lines = {}
    read_zrh_files(lines, os.path.join(cancer_folder, 'cc'), cancer=True, view='CC')
    read_zrh_files(lines, os.path.join(cancer_folder, 'mlo'), cancer=True, view='MLO')
    read_zrh_files(lines, os.path.join(healthy_folder, 'cc'), cancer=False, view='CC')
    read_zrh_files(lines, os.path.join(healthy_folder, 'mlo'), cancer=False, view='MLO')
    return lines


def read_inbreast_csv(fname):
    lines = {}
    with open(fname, 'r') as f:
        next(f)  # skip first line
        for line in csv.reader(f, delimiter=';'):
            line = line[2:]  # skip first two (redacted PatientID, Patient Age)
            laterality, view, date, filename, _, birads, mass, *_ = line  # ignore acr and remainder
            parsed = DotMap()
            parsed.dataset = 'inbreast'
            parsed.laterality = laterality
            parsed.view = view
            parsed.year = int(date[:4])
            parsed.semester = int(date[4:])
            parsed.fid = int(filename)
            parsed.birads = int(birads[0])
            if parsed.birads == 3:
                continue
            parsed.cancer = (parsed.birads > 3)  # (mass.strip() == 'X')
            if parsed.fid in lines:
                raise ValueError('fid already exists: ' + str(parsed.fid))
            lines[parsed.fid] = parsed
    return lines


def load_inbreast():
    dicom_path = os.path.join(inbreast_path, 'AllDICOMs')
    xml_path = os.path.join(inbreast_path, 'AllXML')
    csv_path = os.path.join(inbreast_path, 'INbreast_2.csv')
    image_metadata = read_inbreast_csv(csv_path)
    patients = {}
    for fname in os.listdir(dicom_path):
        path = os.path.join(dicom_path, fname)
        if not os.path.isfile(path) or not path.endswith('.dcm') or fname.startswith('.'):
            continue

        fid, patient_id, modality, laterality, view, _ = fname[:-4].split('_')
        if view == 'ML':
            view = 'MLO'
        fid = int(fid)

        # Add to images
        if fid not in image_metadata:
            continue
        cur = image_metadata[fid]
        cur.image_path = path
        assert laterality == cur.laterality
        assert view == cur.view

        # Add mask
        cur.mask_path = os.path.join(xml_path, str(fid) + '.xml')
        if not os.path.isfile(cur.mask_path):
            print('Missing inbreast mask:', cur.mask_path, cur.birads)

        # Add to patients
        if patient_id not in patients:
            cur = DotMap()
            cur.patient_id = patient_id
            cur.image_metadata = {}
            patients[patient_id] = cur
        cur = patients[patient_id]
        cur.image_metadata[fid] = image_metadata[fid]

    return image_metadata, patients


def read_bcdr_img_csv(fname):
    lines = {}
    with open(fname, 'r') as f:
        next(f)  # skip first line
        for i, line in enumerate(csv.reader(f, delimiter=',')):
            line = [el.strip() for el in line]
            patient_id, study_id, series, image_filename, image_type_name, image_type_id, age, density = line
            parsed = DotMap()
            parsed.dataset = 'bcdr'
            parsed.patient_id = int(patient_id)
            parsed.study_id = int(study_id)
            parsed.series = int(series)
            parsed.image_filename = image_filename
            parsed.laterality = image_type_name[0]
            parsed.view = image_type_name[1:]
            parsed.image_type_id = int(image_type_id)
            parsed.age = int(age)
            parsed.density = None if density == 'NaN' else int(density)
            parsed.mask_path = None

            parsed.fid = i
            if parsed.fid in lines:
                raise ValueError('fid already exists: ' + str(parsed.fid))
            lines[parsed.fid] = parsed
    return lines


def read_bcdr_outlines_csv(fname, images):
    for img in images.values():
        img.cancer = False

    with open(fname, 'r') as f:
        next(f)  # skip first line
        for i, line in enumerate(csv.reader(f, delimiter=',')):
            line = [el.strip() for el in line]
            patient_id, study_id, series, lesion_id, segmentation_id, image_view, image_filename, \
                lw_x_points, lw_y_points, mammography_type, mammography_nodule, mammography_calcification, \
                mammography_microcalcification, mammography_axillary_adenopathy, \
                mammography_architectural_distortion, mammography_stroma_distortion, \
                age, density, classification = line

            for key, img in images.items():
                if img.image_filename == image_filename:
                    classification = classification.strip().lower()
                    img.cancer = (classification == 'malign')
                    img.mask_path = (lw_x_points, lw_y_points)

    return images


def load_bcdr(dataset_name):
    dataset_folder = os.path.join(bcdr_path, "{}_dataset".format(dataset_name))
    csv_path = os.path.join(dataset_folder, "{}_img.csv".format(dataset_name.replace(r'-', '_').lower()))
    outline_path = os.path.join(dataset_folder, "{}_outlines.csv".format(dataset_name.replace(r'-', '_').lower()))

    image_metadata = read_bcdr_img_csv(csv_path)
    if os.path.exists(outline_path):
        image_metadata = read_bcdr_outlines_csv(outline_path, image_metadata)

    patients = {}
    for img_meta in image_metadata.values():
        path = os.path.join(dataset_folder, img_meta.image_filename)
        if not os.path.isfile(path) or not path.endswith('.tif') or os.path.basename(path).startswith('.'):
            raise ValueError("File '{}' not valid.".format(path))

        # Add to images
        img_meta.image_path = path

        if img_meta.view == 'O':
            img_meta.view = 'MLO'

        # Add to patients
        patient_id = img_meta.patient_id
        if patient_id not in patients:
            cur = DotMap()
            cur.patient_id = patient_id
            cur.image_metadata = {}
            patients[patient_id] = cur
        cur = patients[patient_id]
        cur.image_metadata[img_meta.fid] = img_meta

    return image_metadata, patients
