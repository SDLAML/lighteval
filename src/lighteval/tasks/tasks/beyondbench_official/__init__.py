"""The OFFICIAL BeyondBench easy-suite code, vendored (owner 11-18: BeyondBench as a base task). Not a task module (no main.py): `lighteval.tasks.tasks.beyondbench` uses it.

Source: PyPI `beyondbench` 0.2.1 (wheel beyondbench-0.2.1-py3-none-any.whl, sha256 798f741df27b510c17ea3682d1c9f09b79518bd661fc4869e0fad359f55014a0, released 2026-04-17), the package of
github.com/ctrl-gaurav/BeyondBench (paper: arXiv 2509.24210, ICLR 2026), License Apache-2.0 (LICENSE in this folder = the wheel's licenses/LICENSE, sha256 c71d239d...).
Why vendored: the package is not installed in llm_env, and `import beyondbench` runs the package __init__ (CLI, model handler, vLLM, torch: 222 s + 55 s on the login node) before a task class can
be used. The 58 files below are the byte-identical official files (sha256 of each = the wheel's RECORD, checked by tests/unit/tasks/test_beyondbench.py): the 44 `tasks/easy/*_task.py`
(prompts, data generators, golds and verdicts), `utils/parsing.py` + `utils/parsing_utils.py` (the answer extraction), `parsers/common.py`, `parsers/list_parsing_helpers.py` and the 10
`parsers/*_parsing.py` the tasks import. Their relative imports resolve unchanged inside this package. NOT vendored, replaced by stand-ins: the `__init__.py` files (the official ones
import the whole package) and `core/base_task.py` (see its docstring). Nothing else of the package is used (no hard / medium suites, no CLI, no model handler).
"""

OFFICIAL_VERSION = "0.2.1"
OFFICIAL_WHEEL_SHA256 = "798f741df27b510c17ea3682d1c9f09b79518bd661fc4869e0fad359f55014a0"
OFFICIAL_LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
# path inside the package (relative to this folder) -> sha256 of the official file (wheel RECORD)
OFFICIAL_FILES = {
    "parsers/common.py": "a7e8c563f8d40702fddb62997e348275832f9e8aa225a407c28a0efd3b90262b",
    "parsers/cumulative_sum_parsing.py": "bfda1ef68055bcd096b1aba48af92ea76f38a1dfe3c8d38c94b9831d5ab810f3",
    "parsers/element_frequency_parsing.py": "18e6f09d5f364104aaa80a73266a8070cb17c36fdce4f11258225931d6076189",
    "parsers/interleave_lists_parsing.py": "11816eca1127c9c937f075a94a3c053d315704b41b353fd73ead7245360317de",
    "parsers/list_parsing_helpers.py": "5702b84e776d68845ebf4c7b2128ceb8a50a35ba353096ca9e201f4e2880927b",
    "parsers/moving_average_parsing.py": "f10285d19d61f43c242961ef284efd49e2cf6d6373e736fc5da54645df0ad999",
    "parsers/parity_check_parsing.py": "30c3aaa5c0bd4654df402d1da6699750bba9209f4742627bf293e7bd47b6ae11",
    "parsers/reverse_list_parsing.py": "de22d59480de71da3d2ad8e12d993a119d3cb88caf32912f605a8f801112f003",
    "parsers/rotate_list_parsing.py": "5de88ac790506b6e5486a988589392ef357885690063527e2ab6f588cc06f121",
    "parsers/running_average_parsing.py": "67cbbb55b2d78771e7de874fd543e1f207cd00eb872bdfd126a8d3bf4757698e",
    "parsers/set_difference_parsing.py": "b118a2d7714c2ae6ff8fabfd509c0a83f857ca6df0e4d0a5a7afeecc799e34a3",
    "parsers/set_intersection_parsing.py": "fb27925fdbffc70e6c522c896b0577ecae359229ca7ec4904cd4f1a94c241417",
    "tasks/easy/absolute_difference_task.py": "9f9966908a6c25096b2197af1a9f34a7bf8dc256fe714f60679928ded825dfda",
    "tasks/easy/alternating_sum_task.py": "3a375edf172e50d2ac823bb446b21c029e067e2c91063a4b67caa286a770ac95",
    "tasks/easy/comparison_task.py": "806505c4fc1ad3ea8554ab205df83773d3cc66844d49ee3c31f1e07e9fa9fe2f",
    "tasks/easy/count_greater_than_previous_task.py": "e05d8e7523ddc00eb41962aa858708d234ed8b1364874d1ad1c085df0aab0045",
    "tasks/easy/count_multiples_task.py": "8c4c5f06c1a56a2786325d348d3d70d350dbe63eccea5241cc6df82af4df1be4",
    "tasks/easy/count_negative_task.py": "d557e5bff44b7f706c1f79971ed63f693d133cda1984050f67af27d41fa08a86",
    "tasks/easy/count_palindromic_task.py": "28d7d47883e6229129c4170b8f0f1bc1e6ffea38653dd812a875b95990e5da44",
    "tasks/easy/count_perfect_squares_task.py": "f232e9797fea447bf6d767dd3aac9be369f324fd7a236ac2ff446b1383a2f4e4",
    "tasks/easy/count_unique_task.py": "db2888928f7cf8fc83fa32134c6ff045784bfe40667c10fe07d2e610ce240c01",
    "tasks/easy/cumulative_sum_task.py": "5a508f4a1c9af5ab41f7027b51b0ca70fbe252fa5aee873942509408d24fca76",
    "tasks/easy/division_task.py": "af0adcda7c5a640a4b8e352808c8e9a915236a24c0fc41e1e6a6146e1e886996",
    "tasks/easy/dot_product_task.py": "e47519623671fcb559435137e3accb454b48c1134d20fa526fecfdaeebd508e6",
    "tasks/easy/element_frequency_task.py": "cc06d40fb2448ea907dc7a6734c6b0a624eb2ef4a8662771fd37959ea222cda9",
    "tasks/easy/even_count_task.py": "4421134b26cd9df697b4df906af1c632ba5103454808993094c0d87ac44e38d3",
    "tasks/easy/find_maximum_task.py": "29f5c1d14e8b3c5b667631f2623fc9ca0e0ae69bda81cecaa6b897184b62978c",
    "tasks/easy/find_minimum_task.py": "405deca7503d4874806327603cd939770dc8f2af9ed4de2043bc382066c629af",
    "tasks/easy/index_of_maximum_task.py": "beba6754fafd6bf9a4abba54be18e5185d6b7e053b2ab3d74a797a57dd3f9da2",
    "tasks/easy/interleave_lists_task.py": "c8c45900512f9b3167bca7408d36fb00248bd3ab5fbabc5800ee0ea1ca6dd81b",
    "tasks/easy/local_maxima_count_task.py": "191cdc39bd955aaa5bcc6bc9f676ab30cce8eeb740cfcabc0775af98b63f3f83",
    "tasks/easy/longest_increasing_subsequence_task.py": "621f8420baf491ce58733c4f872baf10eccc7c65af2c6316ad25712cfbd8be13",
    "tasks/easy/max_adjacent_difference_task.py": "305de2b3d501a9a8fdb4e311af8d9964b984438b2d12ed28f8af778c656ca5a9",
    "tasks/easy/mean_task.py": "733c33f5c317a122a57dbd88466018878695eb4d4960afcbf8f3aa743c8040b5",
    "tasks/easy/median_task.py": "6e438cd8fd4fcca9501368f401d9232a283d51a4e8788e7ae4f3815030af7491",
    "tasks/easy/mode_task.py": "0abf98d7cde0670867a0402155952b7142f3d7917bd5876dbf0cebd9b965394e",
    "tasks/easy/moving_average_task.py": "1ec635b168c09b666aa3f8088ffe0ec6b3e205e65ecd9ac71a3928016bbd4bb0",
    "tasks/easy/multiplication_task.py": "184dace4c5a37d3d6beb0f5f0491d22395fc3093ac9cf7909a01d27c070c2f2c",
    "tasks/easy/odd_count_task.py": "82d494877ad3e7fdab8d52b81d45b6315ac2904886ca74d923556d15887c5435",
    "tasks/easy/parity_check_task.py": "3c3974acaebceaffbe5576b49ff4ba4d5da5ac84ce004a8df1d9be0fde0d31e4",
    "tasks/easy/range_task.py": "350858faff430bc070c4623c58506db78bccbb6ce88e22af486781f9e83f17d7",
    "tasks/easy/reverse_list_task.py": "f7325451c4eca0a71e8dfe354b7c6d160f0af655d8016cc6131b96531cb365ca",
    "tasks/easy/rotate_list_task.py": "ce4c81435358828c18c07bf2059ae43c030bd279f6ff9a0b9c53bb412e574a73",
    "tasks/easy/running_average_task.py": "c3e3dc630c8c89e1f536319d2872929ffc500524a731e20ee4209c29dbe1c790",
    "tasks/easy/second_maximum_task.py": "842c6cb2b0b513c47fd8ca5f1256cd4f7c4eab43c1f36793f439ade8148b75fe",
    "tasks/easy/second_minimum_task.py": "c51b789152266d41e1ec5b5f514f2819f8161438f0c56c909cf24bb48b3ecdd4",
    "tasks/easy/set_difference_task.py": "8505d0ce0f7f96314a3795bfcdfa78af462a1ad4f580708a3f09a2505e83827d",
    "tasks/easy/set_intersection_task.py": "560838c95fb1a72f190f4b1cd9de2ffb34bef237fdc07113921170ecc5829331",
    "tasks/easy/sorting_task.py": "48bc196c7ad97837ab4be49c9b00c057151f7f681e23d85d89719a9c31352c98",
    "tasks/easy/standard_deviation_task.py": "0e1ee95f51227d9a4a09b3747200fc5014b0780d0fde52241e7986689e069c59",
    "tasks/easy/subtraction_task.py": "85059e7a214a847abebc6d94a52933914ce9c7be04d23b3e90c7335d655d063c",
    "tasks/easy/sum_of_digits_task.py": "c9d6433d428c36aa16606366db5e579dfc143ae4a844cb9ca37cff9b9e63535b",
    "tasks/easy/sum_of_max_indices_task.py": "1b457569e6599f67676427a473a5e5c8552ee9079713917efc4ee2b57ef4abc1",
    "tasks/easy/sum_task.py": "4a49b7cd7b205a4d0cd5d039e50e7d0f44f337f8e72595acbb956edfb1f844dd",
    "tasks/easy/variance_task.py": "8f3a98493d0918f281067e7caebd784b3f6b890eee5e1c2d5397e3e5c515d760",
    "tasks/easy/weighted_sum_task.py": "2f989077dc0cef284f44ae7938ae749ecc5b15603156a7b7c0cd3ce311a6ce73",
    "utils/parsing.py": "7e550a819cd30ee8675eb7cc031244213f8a3481094d806e3a9c8c6dcbfaaa00",
    "utils/parsing_utils.py": "ec71f2dbc37c356c91b464ce51441dfd743763b0bdb9eadb64f88dd572692888",
}
