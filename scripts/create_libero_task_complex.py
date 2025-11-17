"""This is a standalone file for create a task in libero."""
import numpy as np

from libero.libero.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import (
    register_task_info,
    get_task_info,
    generate_bddl_from_task_info,
)


@register_mu(scene_type="kitchen")
class KitchenScene(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "flat_stove": 1,
        }

        object_num_info = {
            "alphabet_soup": 1,
            "tomato_sauce": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.1, 0.15],
                region_name="operation_region",
                target_name=self.workspace_name,
                region_half_len=[0.1, 0.05],
            )
        )
        
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.2, -0.15],
                region_name="stove_region",
                target_name=self.workspace_name,
                region_half_len=[0.1, 0.05],
            )
        )
        
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "tomato_sauce_1", "kitchen_table_operation_region"),
            ("On", "alphabet_soup_1", "kitchen_table_operation_region"),
            ("On", "flat_stove_1", "kitchen_table_stove_region"),
            ("Turnon", "flat_stove_1"),
        ]
        return states


def main():
    objects_of_interest = [
        "tomato_sauce_1",
        "alphabet_soup_1",
    ]
    scene_name = "kitchen_scene"
    register_task_info(
        "Heat the tomato sauce on the stove for 1 second and then heat the alphabet soup",
        scene_name=scene_name,
        objects_of_interest=objects_of_interest,
        goal_states=[
            ("CloseXY", "tomato_sauce_1"),
            ("On", "alphabet_soup_1", "flat_stove_1_cook_region"),
        ],
    )

    bddl_file_names, failures = generate_bddl_from_task_info("./libero/libero/bddl_files/libero_stove")
    print(bddl_file_names)


if __name__ == "__main__":
    main()
