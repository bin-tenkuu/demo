package demo.auth.controller;

import demo.auth.entity.SysMenu;
import demo.auth.entity.SysRole;
import demo.auth.mapper.SysRoleMenuMapper;
import demo.auth.mapper.SysUserRoleMapper;
import demo.auth.model.auth.GrantVo;
import demo.auth.model.auth.SysRoleQuery;
import demo.auth.repository.SysRoleRepository;
import demo.auth.util.SecurityUtils;
import demo.core.model.RequestModel;
import demo.core.model.ResultModel;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import lombok.RequiredArgsConstructor;
import lombok.val;
import org.springframework.web.bind.annotation.*;

import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;

/// @author bin
/// @since 2023/05/30
@Tag(name = "系统角色维护", description = "系统角色维护相关接口")
@RestController
@RequestMapping("/SysRole")
@RequiredArgsConstructor
public class SysRoleController {
    private final SysRoleRepository sysRoleRepository;
    private final SysUserRoleMapper sysUserRoleMapper;
    private final SysRoleMenuMapper sysRoleMenuMapper;

    @Operation(summary = "系统角色列表")
    @PostMapping("/list")
    public ResultModel<List<SysRole>> listSysRole(@RequestBody @NotNull @Valid SysRoleQuery vo) {
        return ResultModel.success(sysRoleRepository.page(vo.toPage(), vo.toQuery()));
    }

    @Operation(summary = "系统角色列表")
    @PostMapping("/list/{menuId}")
    public ResultModel<List<SysRole>> listSysRole(
            @RequestBody @NotNull @Valid SysRoleQuery vo,
            @PathVariable Long menuId
    ) {
        return ResultModel.success(sysRoleRepository.page(vo.toPage(), menuId, vo.toQuery()));
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
            return ResultModel.fail("新增角色'" + sysRole.getRoleName() + "'失败，角色权限已存在");
        }
        sysRoleRepository.updateById(sysRole);
        return ResultModel.success();
    }

    @Operation(summary = "删除系统角色")
    @PostMapping("/delete")
    public ResultModel<?> deleteSysRole(@RequestBody @NotNull @Valid RequestModel<List<Long>> model) {
        val ids = new HashSet<>(model.getData());
        if (ids.isEmpty()) {
            return ResultModel.success();
        }
        val list = sysRoleRepository.query()
                .select(SysRole.ID)
                .eq(SysRole.STATUS, "1")
                .in(SysRole.ID, ids)
                .list()
                .stream().map(SysRole::getId)
                .collect(Collectors.toList());
        if (!list.isEmpty()) {
            // 已经停用的角色可以直接删除
            sysUserRoleMapper.deleteByRoleId(list);
            for (Long id : list) {
                sysRoleMenuMapper.deleteByRoleId(id);
                ids.remove(id);
            }
            sysRoleRepository.removeBatchByIds(list);
        }
        if (!ids.isEmpty()) {
            // 批量停用
            sysRoleRepository.update()
                    .set(SysRole.STATUS, "1")
                    .set(SysRole.UPDATE_BY, SecurityUtils.getUsername().get())
                    .in(SysRole.ID, ids)
                    .update();
        }
        return ResultModel.success();
    }

    @Operation(summary = "根据角色编号获取菜单")
    @PostMapping("/menus/{roleId}")
    public ResultModel<List<SysMenu>> listRolesByUserId(@PathVariable Long roleId) {
        val roles = sysRoleMenuMapper.selectMenuByRoleId(roleId);
        return ResultModel.success(roles);
    }

    @Operation(summary = "批量授权菜单", description = "id为角色id,ids为菜单id")
    @PostMapping("/grantMenu")
    public ResultModel<?> grantMenuToRole(@RequestBody @NotNull @Valid GrantVo vo) {
        sysRoleMenuMapper.deleteByRoleId(vo.getId());
        val menuIds = vo.getTargetIds();
        if (menuIds != null && !menuIds.isEmpty()) {
            sysRoleMenuMapper.insertRoleMenu(vo.getId(), menuIds);
            return ResultModel.success();
        }
        return ResultModel.success();
    }

    @Operation(summary = "批量取消授权菜单", description = "id为角色id,ids为菜单id")
    @PostMapping("/revokeMenu")
    public ResultModel<?> revokeMenuFromRole(@RequestBody @NotNull @Valid GrantVo vo) {
        sysRoleMenuMapper.deleteByRoleMenu(vo.getId(), vo.getTargetIds());
        return ResultModel.success();
    }

    @Operation(summary = "批量授权用户", description = "id为用户id,ids为角色id")
    @PostMapping("/grantUser")
    public ResultModel<?> grantRoleToUser(@RequestBody @NotNull @Valid GrantVo vo) {
        sysUserRoleMapper.insertUsersRole(vo.getId(), vo.getTargetIds());
        return ResultModel.success();
    }

    @Operation(summary = "批量取消授权用户", description = "id为角色id,ids为用户id")
    @PostMapping("/revokeUser")
    public ResultModel<?> revokeRoleFromUser(@RequestBody @NotNull @Valid GrantVo vo) {
        sysUserRoleMapper.deleteByUsersRole(vo.getId(), vo.getTargetIds());
        return ResultModel.success();
    }

}
