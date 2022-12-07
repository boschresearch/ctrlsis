import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from torchvision.utils import save_image

from oasis_tools.utils import recursive_check
from utilfunctions import min_max_scaling, pick_labels, random_string


class ComponentNet(nn.Module):
    def __init__(
        self, num_classes=151, z_dim=64, num_k=5, num_oasis_layers=1, z_as_input=False
    ):
        """Initializes the MLP Direction Model ComponentNet with a fowards pass that delivers a direction
        in dependence of c, k and optionally z.
        Given a latent code z, the mlp predicts the direction v.
        The latent can then be shifted: z_new = z + v.
        The direction v depends on c and k, or on c, k and z: v(c,k) or v(c,k,z).
        The direction v can be computed to be the same for all layers (size c,k,dim_z)
        or different per generator layer (size c,k,dim_z*num_oasis_layers).

        The architecture of the MLP is still experimental, and incoporates:
        - skip connections
        - conditional layers

        Args:
            num_classes (int): Number of classes in the dataset. Defaults to 151 for ADE20k dataset.
            z_dim (int): Defaults to 64.
            num_k (int): Number of direction models per class. Defaults to 5.
            num_oasis_layers (int): The number of layers in the generator - in case v is predicted per layer. Defaults to 1.
            z_as_input (bool): If besides c,k - the direction models also takes z as input.
                Allows for Non-Linear Direction Models. Defaults to False.
        """
        super().__init__()
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.num_oasis_layers = num_oasis_layers
        self.num_k = num_k
        self.z_as_input = z_as_input
        if self.z_as_input:
            input_dim = self.num_classes + self.num_k + self.z_dim
        else:
            input_dim = self.num_classes + self.num_k

        self.first_layer = nn.Sequential(nn.Linear(input_dim, 256), nn.LeakyReLU())
        self.model = nn.ModuleList(
            [
                nn.Linear(256 + input_dim, 256),
                nn.Linear(256 + input_dim, 256),
                nn.Linear(256 + input_dim, 256),
                nn.Linear(256 + input_dim, 256),
                nn.Linear(256 + input_dim, 256),
                nn.Linear(256 + input_dim, 256),
            ]
        )
        self.lrelu = nn.LeakyReLU()
        self.last_layer = nn.Linear(256, self.num_oasis_layers * self.z_dim)

    def forward(self, label, z, k_index):
        batch_size = z.size(0)
        y = F.one_hot(label, self.num_classes).float()
        k = F.one_hot(k_index, self.num_k).float()
        if self.z_as_input:
            input = torch.cat((y, k, z), dim=1)
        else:
            input = torch.cat((y, k), dim=1).float()

        output = self.first_layer(input)
        for _, layer in enumerate(self.model):
            layer_input = torch.cat((output, input), dim=1)
            output = output + self.lrelu(layer(layer_input))
        output = self.last_layer(output)
        output = output.view(batch_size, self.num_oasis_layers * self.z_dim)
        return output


class CombiModel(nn.Module):
    def __init__(
        self,
        generator,
        batch_size,
        z_dim=64,
        num_classes=151,
        k=5,
        c=None,
        num_layers=7,
        model_type="param_ck",
        pred_all_layers=False,
        use_class_mask=False,
        unit_norm=False,
        normalize_images=False,
        alpha_scaling=False,
        feature_name="x",
        avg_l2_norm=None,
        norm_type="direct",
        not_class_specific=False,
        learn_per_layer_weights=False,
        alpha_flipping=False,
        merge_classes=False,
        rotation_alpha=None,
        translation_alpha=None,
        zzAzv_alpha=None,
        flip_zzAzv=False,
    ):
        """Main direction model class with utility methods.

        Args:
            oasis (torch.nn.Module): OASIS Model
            batch_size (int): batch size
            z_dim (int, optional): OASIS Input noise dimension. Defaults to 64.
            num_classes (int, optional): Number of classes (c) supported. Defaults to 151 for ADE dataset.
            c (int, optional): Just consider top c classes by average pixel count on dataset. Defaults to None.
            k (int, optional): Number of direction models per class. Defaults to 5.
            num_layers (int, optional): Number of generator layers (resnet blocks). Defaults to 7.
            model_type (str, optional): See arguments parser for documentation. [choices: param_ck, net_ck, net_ckz]. Defaults to 'param_ck'.
            pred_all_layers (bool, optional): Flags if the should take all the layers for optimization. Defaults to False.
                Rationale: learning a direction means a direction per layer.
            use_class_mask (bool, optional): If spatial sampling. Otherwise global. Defaults to False.
            unit_norm (bool, optional): Normalize the directions. Defaults to False.
            normalize_images (bool, optional): Normalize images. Defaults to False.
            alpha_scaling (bool, optional): Flags if should multiply directions by an scale alpha factor between -3 and 3. Defaults to False.
        """
        super().__init__()

        self.k = k
        self.c = c
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.pred_all_layers = pred_all_layers
        self.layer_multiplier = self.num_layers if self.pred_all_layers else 1
        self.use_class_mask = use_class_mask
        self.normalize_images = normalize_images
        self.alpha_scaling = alpha_scaling
        self.alpha_flipping = alpha_flipping
        self.rotation_alpha = rotation_alpha
        self.translation_alpha = translation_alpha
        self.z_dim = z_dim
        self.generator = generator
        self.feature_name = feature_name
        self.zzAzv_alpha = zzAzv_alpha
        self.return_layers = (
            [self.feature_name]
            if not isinstance(self.feature_name, list)
            else self.feature_name
        )
        self.flip_zzAzv = flip_zzAzv
        self.model_type = model_type
        self.unit_norm = unit_norm
        self.avg_l2_norm = avg_l2_norm
        self.norm_type = norm_type
        self.not_class_specific = not_class_specific
        if self.not_class_specific:
            self.num_classes = 1
        self.learn_per_layer_weights = False  

        
        
        self.k_distribution = Categorical(probs=torch.ones((batch_size, k)) / k)

        if self.model_type == "param_ck":
            self.learned_directions = torch.randn(
                self.num_classes, self.k, self.z_dim * self.layer_multiplier
            )
            self.learned_directions = nn.parameter.Parameter(
                self.learned_directions, requires_grad=True
            )
            self.register_parameter("learned_directions", self.learned_directions)
            recursive_check(self.learned_directions, " self.learned_directions ")

        if self.model_type == "Az_plus_v" or self.model_type == "zzAzv":
            self.learned_directions = torch.randn(
                self.num_classes, self.k, self.z_dim * self.layer_multiplier
            )
            self.learned_directions = nn.parameter.Parameter(
                self.learned_directions, requires_grad=True
            )
            self.register_parameter("learned_directions", self.learned_directions)

            self.learned_matrix = torch.empty(
                self.num_classes, self.k, self.z_dim, self.z_dim
            )
            for c in range(self.num_classes):
                for k in range(self.k):
                    self.learned_matrix[c, k] = torch.nn.init.orthogonal_(
                        self.learned_matrix[c, k]
                    )
                    
                    
                    
                    while torch.slogdet(self.learned_matrix[c, k])[0] < 0.0:
                        self.learned_matrix[c, k] = torch.nn.init.orthogonal_(
                            self.learned_matrix[c, k]
                        )
                        

            self.learned_matrix = nn.parameter.Parameter(
                self.learned_matrix, requires_grad=True
            )
            self.register_parameter("learned_matrix", self.learned_matrix)

        elif self.model_type == "net_ck":
            self.direction_net = ComponentNet(
                num_classes=self.num_classes,
                z_dim=z_dim,
                num_k=k,
                num_oasis_layers=self.layer_multiplier,
            )
        elif self.model_type == "net_ckz":
            self.direction_net = ComponentNet(
                num_classes=self.num_classes,
                z_dim=z_dim,
                num_k=k,
                num_oasis_layers=self.layer_multiplier,
                z_as_input=True,
            )
            self.direction_net.train()

    def adjust_vector_length(self, direction, alpha=None):
        batch_size = direction.size(0)
        len_dir = torch.linalg.norm(direction, ord=2, dim=2)  
        len_dir = len_dir.view(batch_size, self.num_layers, 1)
        
        if alpha is not None:
            alpha = self.avg_l2_norm
        direction = direction / len_dir * alpha

        if self.learn_per_layer_weights:
            direction = direction * self.per_layer_weights.view(-1, 1)

        return direction

    def get_direction(self, k, c, alpha=1.0, z=None):
        return self.learned_directions[c, k]

    def get_random_directions_3d(
        self,
        label_map: torch.Tensor,
        alpha: float = 1.0,
        z: torch.Tensor = None,
        use_all_labels=True,
        c: torch.Tensor = None,
        mode: str = "random_k_per_labelmap_and_class",
        rand_k_per_class: bool = True,
        return_selection: bool = False,
        selection: torch.tensor = None,
    ) -> torch.Tensor:
        """Returns 3D tensor of directions, where each class in each label map
        has a random direction k.

        Note: currently only works for batch size one.

        Args:
            label_map (torch.Tensor): Label map of size (1, C, H, W)
            alpha (float, optional): Scale of direction - keep it at 1.0, since directions are scaled elsewhere.
            Here for compatibility with the get_direction method of other models. Defaults to 1.0.
            z (torch.Tensor, optional): Initial z. Here for compatibility with the get_direction method of other models. Defaults to None.
            use_all_labels (bool, optional): Apply direction k (which is different for each label) to each label.
            If true, c is ignored. Defaults to False.
            mode (str): choices = [ "random_k_per_labelmap_and_class",  "random_k_per_labelmap"],
            rand_k_per_class (bool) = just here for API compatibility with related work

        Returns:
            [torch.Tensor]: noise direction tensor of size (1, D, H, W)

        """

        B, _, H, W = label_map.size()
        batchwise_learned_directions = self.learned_directions.unsqueeze(0).expand(
            B, -1, -1, -1
        )
        _, C, K, D = batchwise_learned_directions.size()
        

        if mode == "random_k_per_labelmap":
            
            
            batchwise_learned_directions = batchwise_learned_directions.permute(
                0, 2, 1, 3
            ).contiguous()
            batchwise_learned_directions = batchwise_learned_directions.view(
                B * K, C, D
            )
            if selection is None:
                
                selection = torch.randint(
                    K, size=(B,), device=label_map.device
                )  
            selection_bool = F.one_hot(selection, K).view(B * K).bool()
            batchwise_directions = batchwise_learned_directions[
                selection_bool
            ]  

        elif mode == "random_k_per_labelmap_and_class":
            
            
            
            if selection is None:
                selection = torch.randint(
                    K, size=(B * C,), device=label_map.device
                )  

            
            selection_bool = (
                F.one_hot(selection, K)
                .view(
                    B * C * K,
                )
                .bool()
            )
            
            batchwise_learned_directions = (
                batchwise_learned_directions.contiguous().view(B * C * K, D)
            )
            batchwise_directions = batchwise_learned_directions[selection_bool]
            batchwise_directions = batchwise_directions.view(B, C, D)

        
        label_map_rolled_up = (
            label_map.permute(0, 2, 3, 1)
            .contiguous()
            .view(
                B * H * W * C,
            )
            .bool()
        )  
        batchwise_directions = batchwise_directions.view(B, 1, 1, C, D)
        batchwise_directions = batchwise_directions.expand(
            -1, H, W, -1, -1
        ).contiguous()
        
        batchwise_directions = batchwise_directions.view(B * H * W * C, D)
        direction_map_3D = batchwise_directions[label_map_rolled_up]

        
        
        
        
        
        
        

        
        
        
        direction_map_3D = direction_map_3D.view(B, H, W, D)
        direction_map_3D = direction_map_3D.permute(
            0, 3, 1, 2
        ).contiguous()  

        
        

        
        
        
        
        

        if not use_all_labels:
            
            
            assert c is not None, "c must be an int that specifies a class"
            direction_map_3D = (label_index_map == c) * direction_map_3D

        if return_selection:
            return direction_map_3D, selection
        else:
            return direction_map_3D

    def get_direction_batched(self, k, c, alpha=1.0, z=None, return_matrix=False):
        
        assert (
            self.learned_directions.size(1) >= k.max()
        ), f"selected k={k} exceed learned num k: {self.learned_directions.size()}"
        c = c.long()
        
        
        if return_matrix:
            return {"v": self.learned_directions[c, k], "A": self.learned_matrix[c, k]}
        else:
            return self.learned_directions[c, k]

    
    def forward(
        self,
        z,
        z2,
        label,
        k=None,
        chosen_label=None,
        allow_scaling=False,
        symmetric_directions=False,
        optim_classes=None,
        stochastic_triplet=False,
        mirror_triplet=False,
        multi_triplet=False,
    ):  
        batch_size = label.size(0)
        H = label.size(2)
        W = label.size(3)

        if chosen_label is None:
            if self.not_class_specific:
                chosen_label = torch.zeros(
                    label.size(0), device=label.device, dtype=torch.long
                )
            else:
                
                chosen_label = (
                    pick_labels(label, self.c, optim_classes).long().to(label.device)
                )

        
        if k is None:
            k = self.k_distribution.sample().to(label.device)[:batch_size]

        
        k_probs = (
            torch.ones((batch_size, self.k), device=label.device)
            / self.k
            * (1 - F.one_hot(k, self.k).float())
        )
        k2 = Categorical(probs=k_probs).sample().to(label.device)[:batch_size]

        if self.translation_alpha is not None:
            alpha = self.translation_alpha
        elif allow_scaling and self.alpha_scaling:
            alpha = torch.rand(1, device=label.device) * 6 - 3
        else:
            alpha = 1.0

        if allow_scaling and self.alpha_flipping:
            factor = ((torch.rand(1, device=label.device) > 0.5).float() * 2) - 1
            
            alpha = alpha * factor  

        if self.rotation_alpha is not None:
            rotation_alpha = self.rotation_alpha
        else:
            rotation_alpha = 1.0

        
        
        

        if self.model_type == "param_ck":

            if multi_triplet:

                direction, selection = self.get_random_directions_3d(
                    label_map=label,
                    alpha=1.0,
                    z=z,  
                    c=None,  
                    use_all_labels=True,
                    rand_k_per_class=True,
                    return_selection=True,
                )  

                
                

                
                k_probs = (
                    torch.ones((selection.size(0), self.k), device=label.device)
                    / self.k
                    * (1 - F.one_hot(selection, self.k).float())
                )
                
                selection2 = Categorical(probs=k_probs).sample()  

                
                
                

                
                direction2 = self.get_random_directions_3d(
                    label_map=label,
                    alpha=1.0,
                    z=z,  
                    c=None,  
                    use_all_labels=True,
                    rand_k_per_class=True,
                    selection=selection2,
                )  

                
                direction = direction.unsqueeze(1).expand(
                    -1, self.num_layers, -1, -1, -1
                )
                direction2 = direction2.unsqueeze(1).expand(
                    -1, self.num_layers, -1, -1, -1
                )

                
                area_direction = self.learned_directions[chosen_label, k]

                if self.pred_all_layers:
                    area_direction = area_direction.view(
                        batch_size, self.num_layers, self.z_dim
                    )
                else:
                    area_direction = area_direction.view(batch_size, 1, self.z_dim)
                    area_direction = area_direction.expand(
                        batch_size, self.num_layers, -1
                    )  

            else:

                direction = self.learned_directions[chosen_label, k]
                direction2 = self.learned_directions[chosen_label, k2]

                if self.pred_all_layers:
                    direction = direction.view(batch_size, self.num_layers, self.z_dim)
                    direction2 = direction2.view(
                        batch_size, self.num_layers, self.z_dim
                    )
                else:
                    direction = direction.view(batch_size, 1, self.z_dim)
                    direction = direction.expand(batch_size, self.num_layers, -1)

                    direction2 = direction2.view(batch_size, 1, self.z_dim)
                    direction2 = direction2.expand(
                        batch_size, self.num_layers, -1
                    )  

            if self.unit_norm and self.norm_type == "indirect":
                direction = self.adjust_vector_length(direction)
                direction2 = self.adjust_vector_length(direction2)

        if self.model_type == "Az_plus_v" or self.model_type == "zzAzv":
            

            matrix = self.learned_matrix[chosen_label, k]  
            matrix2 = self.learned_matrix[chosen_label, k2]  

            direction = self.learned_directions[chosen_label, k]
            direction2 = self.learned_directions[chosen_label, k2]

            if self.pred_all_layers:
                direction = direction.view(batch_size, self.num_layers, self.z_dim)
                direction2 = direction2.view(batch_size, self.num_layers, self.z_dim)
            else:
                direction = direction.view(batch_size, 1, self.z_dim)
                direction = direction.expand(batch_size, self.num_layers, -1)

                direction2 = direction2.view(batch_size, 1, self.z_dim)
                direction2 = direction2.expand(
                    batch_size, self.num_layers, -1
                )  

                matrix = matrix.view(batch_size, 1, self.z_dim, self.z_dim)
                matrix = matrix.expand(batch_size, self.num_layers, -1, -1).contiguous()

                matrix2 = matrix2.view(batch_size, 1, self.z_dim, self.z_dim)
                matrix2 = matrix2.expand(
                    batch_size, self.num_layers, -1, -1
                ).contiguous()  

        if stochastic_triplet:
            
            
            bs = direction.size(0)
            alpha1 = alpha * (
                ((torch.rand(bs, 1, 1, device=label.device) > 0.5).float() * 2) - 1
            )
            alpha2 = alpha * (
                ((torch.rand(bs, 1, 1, device=label.device) > 0.5).float() * 2) - 1
            )
            
            
            
        else:
            alpha1 = alpha
            alpha2 = alpha

        
        per_layer_noise_add_k1_alpha_plus = alpha1 * direction
        per_layer_noise_add_k2_alpha_plus = alpha2 * direction2

        
        
        

        
        
        
        if multi_triplet:
            per_layer_noise_original = (
                z.unsqueeze(2)
                .unsqueeze(2)
                .unsqueeze(1)
                .expand(
                    -1,
                    self.num_layers,
                    -1,
                    H,
                    W,
                )
            )  
            per_layer_noise_original2 = (
                z2.unsqueeze(2)
                .unsqueeze(2)
                .unsqueeze(1)
                .expand(
                    -1,
                    self.num_layers,
                    -1,
                    H,
                    W,
                )
            )  

            
            area_per_layer_noise_original = z.unsqueeze(1).expand(
                -1, self.num_layers, -1
            )  

            area_per_layer_noise_add_k1_alpha_plus = alpha * area_direction
            area_per_layer_noise_shifted = (
                area_per_layer_noise_original + area_per_layer_noise_add_k1_alpha_plus
            )

        else:
            per_layer_noise_original = z.unsqueeze(1).expand(
                -1, self.num_layers, -1
            )  
            per_layer_noise_original2 = z2.unsqueeze(1).expand(
                -1, self.num_layers, -1
            )  

        
        

        
        

        if self.model_type == "param_ck":
            per_layer_noise_shifted = (
                per_layer_noise_original + per_layer_noise_add_k1_alpha_plus
            )  
            per_layer_noise_shifted_neg = (
                per_layer_noise_original + per_layer_noise_add_k2_alpha_plus
            )  
            per_layer_noise_shifted_pos = (
                per_layer_noise_original2 + per_layer_noise_add_k1_alpha_plus
            )  
        elif self.model_type == "Az_plus_v" or self.model_type == "zzAzv":

            num_b, num_layer, z_dim = per_layer_noise_original.size()
            per_layer_noise_affine = torch.bmm(
                rotation_alpha * matrix.view(num_b * num_layer, z_dim, z_dim),
                per_layer_noise_original.reshape(num_b * num_layer, z_dim, 1),
            )
            per_layer_noise_affine = per_layer_noise_affine.view(
                num_b, num_layer, z_dim
            )

            per_layer_noise_affine2 = torch.bmm(
                rotation_alpha * matrix2.view(num_b * num_layer, z_dim, z_dim),
                per_layer_noise_original.reshape(num_b * num_layer, z_dim, 1),
            )
            per_layer_noise_affine2 = per_layer_noise_affine2.view(
                num_b, num_layer, z_dim
            )

            per_layer_noise_shifted = (
                per_layer_noise_affine + per_layer_noise_add_k1_alpha_plus
            )  
            per_layer_noise_shifted_neg = (
                per_layer_noise_affine + per_layer_noise_add_k2_alpha_plus
            )  
            per_layer_noise_shifted_pos = (
                per_layer_noise_affine2 + per_layer_noise_add_k1_alpha_plus
            )  

            if self.model_type == "zzAzv":
                per_layer_noise_shifted = self.adjust_vector_length(
                    per_layer_noise_shifted, 1.0
                )
                per_layer_noise_shifted_neg = self.adjust_vector_length(
                    per_layer_noise_shifted_neg, 1.0
                )
                per_layer_noise_shifted_pos = self.adjust_vector_length(
                    per_layer_noise_shifted_pos, 1.0
                )

                if self.flip_zzAzv:
                    bs = direction.size(0)
                    zzAzv_alpha1 = alpha * (
                        ((torch.rand(bs, 1, 1, device=label.device) > 0.5).float() * 2)
                        - 1
                    )
                    zzAzv_alpha2 = alpha * (
                        ((torch.rand(bs, 1, 1, device=label.device) > 0.5).float() * 2)
                        - 1
                    )
                else:
                    zzAzv_alpha1 = self.zzAzv_alpha
                    zzAzv_alpha2 = self.zzAzv_alpha

                per_layer_noise_shifted = (
                    per_layer_noise_original + zzAzv_alpha1 * per_layer_noise_shifted
                )
                per_layer_noise_shifted_neg = (
                    per_layer_noise_original
                    + zzAzv_alpha2 * per_layer_noise_shifted_neg
                )
                per_layer_noise_shifted_pos = (
                    per_layer_noise_original
                    + zzAzv_alpha1 * per_layer_noise_shifted_pos
                )

            if self.unit_norm and self.norm_type == "indirect":

                print("never execute this - indirect via ortho loss is way better")
                exit()
                
                len1 = torch.linalg.norm(
                    per_layer_noise_affine, ord=2, dim=2, keepdim=True
                )  
                len2 = torch.linalg.norm(
                    per_layer_noise_affine2, ord=2, dim=2, keepdim=True
                )
                per_layer_noise_affine = self.adjust_vector_length(
                    per_layer_noise_affine, alpha=len1 + self.avg_l2_norm
                )
                per_layer_noise_affine2 = self.adjust_vector_length(
                    per_layer_noise_affine2, alpha=len2 + self.avg_l2_norm
                )

        if mirror_triplet:
            per_layer_noise_shifted_neg_neg = (
                per_layer_noise_original - per_layer_noise_add_k2_alpha_plus
            )

        
        
        
        

        features = self.generator(
            label, z, self.return_layers, per_layer_noise=per_layer_noise_original
        )

        if multi_triplet:
            label_to_shift = None
        elif self.use_class_mask:
            label_to_shift = chosen_label
        else:
            label_to_shift = None
        
        features_shifted = self.generator(
            label,
            z,
            self.return_layers,
            per_layer_noise=per_layer_noise_shifted,
            label_to_shift=label_to_shift,  
        )

        if multi_triplet:
            area_features_shifted = self.generator(
                label,
                z,
                self.return_layers,
                per_layer_noise=area_per_layer_noise_shifted,
                label_to_shift=chosen_label if self.use_class_mask else None,
            )

        
        features_shifted_negative = self.generator(
            label,
            z,
            self.return_layers,
            per_layer_noise=per_layer_noise_shifted_neg,
            label_to_shift=label_to_shift,  
        )

        
        features_shifted_positive = self.generator(
            label,
            z2,
            self.return_layers,
            per_layer_noise=per_layer_noise_shifted_pos,
            label_to_shift=label_to_shift,  
        )
        if mirror_triplet:
            features_shifted_negative_negative = self.generator(
                label,
                z,
                self.return_layers,
                per_layer_noise=per_layer_noise_shifted_neg_neg,
                label_to_shift=label_to_shift,  
            )

        outputs = {
            "features": features,
            "features_shifted": features_shifted,
            "features_shifted_negative": features_shifted_negative,
            "features_shifted_positive": features_shifted_positive,
            "image": features["x"],
            "image_shifted": features_shifted["x"],
            "chosen_label": chosen_label,
            "chosen_k": k,
            "chosen_label_img": chosen_label.view(batch_size, 1, 1, 1).expand(
                -1, -1, 100, 100
            ),
        }
        if mirror_triplet:
            outputs[
                "features_shifted_negative_negative"
            ] = features_shifted_negative_negative

        if multi_triplet:
            outputs["area_features_shifted"] = area_features_shifted
        
        
        
        
        
        
        
        
        

        if "mask" in features_shifted:
            outputs["noise_mask"] = features_shifted["mask"]

        return outputs
