from npanalyst import activity


def test_filenames2samples_sees_all_relevant():
    samples = ["Sample1", "Sample2", "Sample3"]
    filelist = "Sample1-1.mzml|Sample2-2.mzml|Sample3-3.mzml"
    expected = ["Sample1", "Sample2", "Sample3"]
    assert (
        activity.filenames2samples(filenames=filelist, all_samples=samples) == expected
    )


def test_filenames2samples_sees_all_relevant():
    samples = ["Sample1", "Sample2", "Sample3"]
    filelist = "Sample1|Sample2|Sample3"
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


def test_filenames2_samples_real_mzmine():
    samples = [
        "RLUS-2048D",
        "RLUS-2048E",
        "RLUS-2059E",
        "RLUS-2067F",
        "RLUS-2071E",
        "RLUS-2110D",
        "RLUS-2130C",
        "RLUS-2132C",
        "RLUS-2135E",
        "RLUS-2136A",
        "RLUS-2155E",
        "RLUS-2155F",
        "RLUS-2156E",
        "RLUS-2157B",
        "RLUS-2158A",
        "RLUS-2159D",
        "RLUS-2177C",
        "RLUS-2180C",
        "RLUS-2181C",
        "RLUS-2181F",
        "RLUS-2184C",
        "RLUS-2184D",
        "RLUS-2184E",
        "RLUS-2186A",
        "RLUS-2186B",
        "RLUS-2189D",
        "RLUS-2191E",
        "RLUS-2193A",
        "RLUS-2194E",
        "RLUS-2199A",
        "RLUS-2208C",
        "RLUS-2213E",
        "RLUS-2213F",
        "RLUS-2214A",
        "RLUS-2214F",
    ]
    filelist = "RLUS-2048D Peak Area|RLUS-2048E Peak Area|RLUS-2059E Peak Area|RLUS-2067F Peak Area|RLUS-2071E Peak Area|RLUS-2110D Peak Area|RLUS-2130C Peak Area|RLUS-2132C Peak Area|RLUS-2135E Peak Area|RLUS-2136A Peak Area|RLUS-2155E Peak Area|RLUS-2155F Peak Area|RLUS-2156E Peak Area|RLUS-2157B Peak Area|RLUS-2158A Peak Area|RLUS-2159D Peak Area|RLUS-2177C Peak Area|RLUS-2180C Peak Area|RLUS-2181C Peak Area|RLUS-2181F Peak Area|RLUS-2184C Peak Area|RLUS-2184D Peak Area|RLUS-2184E Peak Area|RLUS-2186A Peak Area|RLUS-2186B Peak Area|RLUS-2189D Peak Area|RLUS-2191E Peak Area|RLUS-2193A Peak Area|RLUS-2194E Peak Area|RLUS-2199A Peak Area|RLUS-2208C Peak Area|RLUS-2213E Peak Area|RLUS-2213F Peak Area|RLUS-2214A Peak Area|RLUS-2214F Peak Area"
    filelist1 = "RLUS-2048D|RLUS-2048E|RLUS-2059E|RLUS-2067F|RLUS-2071E|RLUS-2110D|RLUS-2130C|RLUS-2132C|RLUS-2135E|RLUS-2136A|RLUS-2155E|RLUS-2155F|RLUS-2156E|RLUS-2157B|RLUS-2158A|RLUS-2159D|RLUS-2177C|RLUS-2180C|RLUS-2181C|RLUS-2181F|RLUS-2184C|RLUS-2184D|RLUS-2184E|RLUS-2186A|RLUS-2186B|RLUS-2189D|RLUS-2191E|RLUS-2193A|RLUS-2194E|RLUS-2199A|RLUS-2208C|RLUS-2213E|RLUS-2213F|RLUS-2214A|RLUS-2214F"
    assert (
        activity.filenames2samples(filenames=filelist, all_samples=samples) == samples
    )
    assert (
        activity.filenames2samples(filenames=filelist1, all_samples=samples) == samples
    )
