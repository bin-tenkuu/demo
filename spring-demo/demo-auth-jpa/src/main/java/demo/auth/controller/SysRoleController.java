package demo.auth.controller;

import demo.auth.entity.SysMenu;
import demo.auth.entity.SysRole;
import demo.auth.entity.SysRoleMenu;
import demo.auth.entity.SysUserRole;
import demo.auth.model.auth.GrantVo;
import demo.auth.model.auth.SysRoleQuery;
import demo.auth.repository.SysRoleMenuRepository;
import demo.auth.repository.SysRoleRepository;
import demo.auth.repository.SysUserRoleRepository;
import demo.auth.util.SecurityUtils;
import demo.core.model.RequestModel;
import demo.core.model.ResultModel;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.persistence.criteria.JoinType;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import lombok.RequiredArgsConstructor;
import lombok.val;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.web.bind.annotation.*;

import java.util.HashSet;
import java.util.List;

/// @author bin
/// @since 2023/05/30
@Tag(name = "系统角色维护", description = "系统角色维护相关接口")
@RestController
@RequestMapping("/SysRole")
@RequiredArgsConstructor
public class SysRoleController {
    private final SysRoleRepository sysRoleRepository;
    private final SysUserRoleRepository sysUserRoleRepository;
    private final SysRoleMenuRepository sysRoleMenuRepository;

    @Operation(summary = "系统角色列表")
    @PostMapping("/list")
    public ResultModel<List<SysRole>> listSysRole(@RequestBody @NotNull @Valid SysRoleQuery vo) {
        return ResultModel.success(sysRoleRepository.findAll(vo, vo.toPage()));
    }

    @Operation(summary = "系统角色列表")
    @PostMapping("/list/{menuId}")
    public ResultModel<List<SysRole>> listSysRole(
            @RequestBody @NotNull @Valid SysRoleQuery vo,
            @PathVariable Long menuId
    ) {
        Specification<SysRole> join = (root, query, cb) -> {
            var roleMenu = root.join("roleMenus", JoinType.INNER);
            return cb.equal(roleMenu.get("id").get("menuId"), menuId);
        };
        return ResultModel.success(sysRoleRepository.findAll(join.and(vo), vo.toPage()));
    }

    @Operation(summary = "新增系统角色")
    @PostMapping("/new")
    public ResultModel<?> newSysRole(@RequestBody @NotNull @Valid SysRole sysRole) {
        if (sysRoleRepository.checkRoleKeyExist(sysRole)) {
            return ResultModel.fail("新增角色'" + sysRole.getRoleName() + "'失败，角色权限已存在");
        }
        sysRoleRepository.save(sysRole);
        return ResultModel.success();
    }

    @Operation(summary = "修改系统角色")
    @PostMapping("/update")
    public ResultModel<?> updateSysRole(@RequestBody @NotNull @Valid SysRole sysRole) {
        if (sysRoleRepository.checkRoleKeyExist(sysRole)) {
            return ResultModel.fail("修改角色'" + sysRole.getRoleName() + "'失败，角色权限已存在");
        }
        sysRoleRepository.save(sysRole);
        return ResultModel.success();
    }

    @Operation(summary = "删除系统角色")
    @PostMapping("/delete")
    public ResultModel<?> deleteSysRole(@RequestBody @NotNull @Valid RequestModel<List<Long>> model) {
        val ids = new HashSet<>(model.getData());
        if (ids.isEmpty()) {
            return ResultModel.success();
        }
        val list = sysRoleRepository.findIdByStatusAndIdIn(1, ids);
        if (!list.isEmpty()) {
            // 已经停用的角色可以直接删除
            sysUserRoleRepository.deleteByRoleIdIn(list);
            list.forEach(ids::remove);
            sysRoleRepository.deleteAllById(list);
        }
        if (!ids.isEmpty()) {
            // 批量停用
            sysRoleRepository.updateAllByIdIn(SecurityUtils.getUsername().orElse(null), ids);
        }
        return ResultModel.success();
    }

    @Operation(summary = "根据角色编号获取菜单")
    @PostMapping("/menus/{roleId}")
    public ResultModel<List<SysMenu>> listRolesByUserId(@PathVariable Long roleId) {
        val roles = sysRoleMenuRepository.findMenuByRoleId(roleId);
        return ResultModel.success(roles);
    }

    @Operation(summary = "批量授权菜单", description = "id为角色id,ids为菜单id")
    @PostMapping("/grantMenu")
    public ResultModel<?> grantMenuToRole(@RequestBody @NotNull @Valid GrantVo vo) {
        var roleId = vo.getId();
        // sysRoleMenuRepository.deleteByRoleId(roleId);
        var list = vo.getTargetIds().stream()
                .map(menuId -> new SysRoleMenu(roleId, menuId))
                .toList();
        sysRoleMenuRepository.saveAll(list);
        return ResultModel.success();
    }

    @Operation(summary = "批量取消授权菜单", description = "id为角色id,ids为菜单id")
    @PostMapping("/revokeMenu")
    public ResultModel<?> revokeMenuFromRole(@RequestBody @NotNull @Valid GrantVo vo) {
        sysRoleMenuRepository.deleteByRoleIdAndMenuIdIn(vo.getId(), vo.getTargetIds());
        return ResultModel.success();
    }

    @Operation(summary = "批量授权用户", description = "id为用户id,ids为角色id")
    @PostMapping("/grantUser")
    public ResultModel<?> grantRoleToUser(@RequestBody @NotNull @Valid GrantVo vo) {
        var userId = vo.getId();
        var list = vo.getTargetIds().stream()
                .map(roleId -> new SysUserRole(userId, roleId))
                .toList();
        sysUserRoleRepository.saveAll(list);
        return ResultModel.success();
    }

    @Operation(summary = "批量取消授权用户", description = "id为角色id,ids为用户id")
    @PostMapping("/revokeUser")
    public ResultModel<?> revokeRoleFromUser(@RequestBody @NotNull @Valid GrantVo vo) {
        sysUserRoleRepository.deleteByRoleIdAndUserIdIn(vo.getId(), vo.getTargetIds());
        return ResultModel.success();
    }

}
