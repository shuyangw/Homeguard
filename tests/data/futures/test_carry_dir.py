from src.data.futures.paths import carry_dir, roll_calendar_dir


def test_carry_dir_sibling_of_roll_calendar():
    assert carry_dir().name == "carry"
    assert carry_dir().parent == roll_calendar_dir().parent
