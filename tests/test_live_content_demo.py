import numpy as np

from scripts.live_content_demo import _next_deadline, build_parser, make_live_card


def test_live_card_encodes_sequence_orientation_and_time():
    card = make_live_card(42, 1.25, 320, 180)

    assert card.shape == (180, 320, 3)
    assert card.dtype == np.uint8
    assert card[0, 0].tolist() == [30, 80, 245]
    assert card[0, -1].tolist() == [40, 220, 60]
    assert card[-1, -1].tolist() == [235, 90, 35]
    assert card[-1, 0].tolist() == [30, 220, 235]
    assert np.count_nonzero(card != card[90, 160]) > 0


def test_live_producer_skips_missed_ticks_instead_of_bursting_to_catch_up():
    assert np.isclose(_next_deadline(10.0, 10.01, 0.1), 10.1)
    assert np.isclose(_next_deadline(10.0, 10.45, 0.1), 10.55)


def test_live_demo_accepts_a_video_producer_at_its_native_rate():
    args = build_parser().parse_args(["--content-video", "interface.mp4"])

    assert args.content_video == "interface.mp4"
    assert args.producer_fps is None
