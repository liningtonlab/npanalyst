from logging import exception
from npanalyst import activity


def test_filenames2samples_sees_all_relevant():
    samples = ["Sample1", "Sample2", "Sample3"]
    filelist = "Sample1-1.mzml|Sample2-2.mzml|Sample3-3.mzml"
    expected = ["Sample1", "Sample2", "Sample3"]
    assert (
        activity.filenames2samples(filenames=filelist, all_samples=samples) == expected
    )


def test_filenames2samples_sees_all_relevant_not_all():
    samples = ["Sample1", "Sample2", "Sample3"]
    filelist = "Sample1-1.mzml|Sample2-2.mzml"
    expected = ["Sample1", "Sample2"]
    assert (
        activity.filenames2samples(filenames=filelist, all_samples=samples) == expected
    )


def test_filenames2samples_sees_all_relevant_is_case_sensitive():
    samples = ["Sample1", "Sample2", "Sample3"]
    filelist = "sample1-1.mzml|Sample2-2.mzml|Sample3-3.mzml"
    expected = ["Sample2", "Sample3"]
    assert (
        activity.filenames2samples(filenames=filelist, all_samples=samples) == expected
    )


def test_filenames2samples_sees_all_relevant_replicates():
    samples = ["Sample1", "Sample2", "Sample3"]
    filelist = "Sample1-1.mzml|Sample1-2.mzml|Sample1-3.mzml|Sample2-1.mzml|Sample2-2.mzml|Sample2-3.mzml"
    expected = ["Sample1", "Sample2"]
    assert (
        activity.filenames2samples(filenames=filelist, all_samples=samples) == expected
    )
