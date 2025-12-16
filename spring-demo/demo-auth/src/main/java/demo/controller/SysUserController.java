package demo.controller;

import demo.entity.SysMenu;
import demo.entity.SysUser;
import demo.mapper.SysUserRoleMapper;
import demo.model.RequestModel;
import demo.model.ResultModel;
import demo.model.auth.GrantVo;
import demo.model.auth.SysUserAuthUpdate;
import demo.model.auth.SysUserNewVo;
import demo.model.auth.SysUserQuery;
import demo.repository.SysMenuRepository;
import demo.repository.SysUserAuthRepository;
import demo.repository.SysUserRepository;
import demo.service.auth.TokenService;
import demo.util.SecurityUtils;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.val;
import org.springframework.web.bind.annotation.*;

import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;

/// @author bin
/// @since 2025/07/15
@Tag(name = "系统用户维护", description = "系统用户维护相关接口")
@RestController
@RequiredArgsConstructor
@RequestMapping("/SysUser")
public class SysUserController {
    private final SysUserRepository sysUserRepository;
    private final SysUserAuthRepository sysUserAuthRepository;
    private final SysMenuRepository sysMenuRepository;
    private final TokenService tokenService;
    private final SysUserRoleMapper sysUserRoleMapper;

    @Operation(summary = "查询系统用户列表")
    @PostMapping("/list")
    public ResultModel<List<SysUser>> list(@RequestBody SysUserQuery query) {
        var sysUserPage = sysUserRepository.page(query.toPage(), query.toQuery());
        return ResultModel.success(sysUserPage);
    }

    @Operation(summary = "查询系统用户列表-根据角色筛选")
    @PostMapping("/list/{roleId}")
    public ResultModel<List<SysUser>> listSysUserWithRole(
            @RequestBody @Valid SysUserQuery vo,
            @PathVariable Long roleId
    ) {
        return ResultModel.success(sysUserRepository.page(vo.toPage(), roleId, vo.toQuery()));
    }

    @Operation(summary = "查询系统用户列表-未分配角色")
    @PostMapping("/list/unallocated")
    public ResultModel<List<SysUser>> unallocatedList(@RequestBody @Valid SysUserQuery vo) {
        return ResultModel.success(sysUserRepository.pageUnAlloced(vo.toPage(), vo.toQuery()));
    }

    @Operation(summary = "新增系统用户")
    @PostMapping("/new")
    public ResultModel<?> newSysUser(@RequestBody @Valid SysUserNewVo vo) {
        var sysUser = new SysUser();
        sysUser.setNickName(vo.getNickName());
        sysUser.setEmail(vo.getEmail());
        sysUser.setPhoneNumber(vo.getPhoneNumber());
        sysUser.setAvatar(vo.getAvatar());
        sysUser.setStatus(vo.getStatus());

        var username = vo.getUsername();
        var password = vo.getPassword();
        tokenService.loginPreCheck(username, password);
        if (!sysUserAuthRepository.checkUserNameExist(username)) {
            return ResultModel.fail("新增用户'" + username + "'失败，账号已存在");
        }
        var encrypt = tokenService.encode(password);
        var sysUserAuth = new demo.entity.SysUserAuth();
        sysUserAuth.setUsername(username);
        sysUserAuth.setPassword(encrypt);
        sysUserAuthRepository.save(sysUserAuth);
        sysUserRepository.save(sysUser);
        val roleIds = vo.getRoleIds();
        if (roleIds != null && !roleIds.isEmpty()) {
            sysUserRoleMapper.insertUserRoles(sysUser.getId(), roleIds);
        }
        return ResultModel.success();
    }

    @Operation(summary = "修改系统用户")
    @PostMapping("/update")
    public ResultModel<?> updateSysUser(@RequestBody @Valid SysUser sysUser) {
        sysUserRepository.updateById(sysUser);
        return ResultModel.success();
    }

    @Operation(summary = "更新密码")
    @PostMapping("/updatePassword")
    public ResultModel<?> updatePassword(@RequestBody @Valid SysUserAuthUpdate update) {
        var auth = sysUserAuthRepository.findByUsername(update.getUsername());
        if (auth == null) {
            return ResultModel.fail("用户不存在");
        }
        if (!update.getType().equals(auth.getType())) {
            return ResultModel.fail("密码类型不匹配");
        }
        if (!SecurityUtils.getUserId().map(auth.getUserId()::equals).orElse(false)) {
            return ResultModel.fail("只能修改自己的密码");
        }
        if (!tokenService.matches(update.getOldPassword(), auth.getPassword())) {
            return ResultModel.fail("旧密码不正确");
        }
        var encode = tokenService.encode(update.getNewPassword());
        auth.setPassword(encode);
        sysUserAuthRepository.updateById(auth);
        return ResultModel.success();
    }

    @Operation(summary = "删除系统用户")
    @PostMapping("/delete")
    public ResultModel<?> deleteSysUser(@RequestBody @Valid RequestModel<List<Long>> model) {
        val ids = new HashSet<>(model.getData());
        if (ids.contains(0L)) {
            return ResultModel.fail("超级管理员不能删除");
        }
        val list = sysUserRepository.query()
                .select(SysUser.ID)
                .eq(SysUser.STATUS, "1")
                .in(SysUser.ID, ids)
                .list()
                .stream().map(SysUser::getId)
                .collect(Collectors.toList());
        if (!list.isEmpty()) {
            // 已经停用的用户可以直接删除
            for (Long id : list) {
                sysUserRoleMapper.deleteByUserId(id);
                ids.remove(id);
            }
            sysUserRepository.removeByIds(list);
        }
        if (!ids.isEmpty()) {
            // 批量停用
            sysUserRepository.update()
                    .set(SysUser.STATUS, "1")
                    .set(SysUser.UPDATE_BY, SecurityUtils.getUsername())
                    .in(SysUser.ID, ids)
                    .update();
        }
        return ResultModel.success();
    }

    @Operation(summary = "根据当前用户编号获取详细信息")
    @GetMapping(value = "/info")
    public ResultModel<SysUser> getInfo() {
        Long userId = SecurityUtils.getUserId().orElseThrow(() -> new RuntimeException("未登陆用户"));
        SysUser sysUser = sysUserRepository.getById(userId);
        return ResultModel.success(sysUser);
    }

    @Operation(summary = "根据当前用户获取菜单")
    @PostMapping("/menus")
    public ResultModel<List<SysMenu>> listMenusByUserId() {
        Long userId = SecurityUtils.getUserId().orElseThrow(() -> new RuntimeException("未登陆用户"));
        val menus = sysMenuRepository.listMenuByUserId(userId);
        return ResultModel.success(menus);
    }

    @Operation(summary = "批量授权角色", description = "id为用户id,ids为角色id")
    @PostMapping("/grantRole")
    public ResultModel<Integer> grantRoleToUser(@RequestBody @Valid GrantVo vo) {
        sysUserRoleMapper.deleteByUserId(vo.getId());
        val roleIds = vo.getTargetIds();
        if (roleIds != null && !roleIds.isEmpty()) {
            val result = sysUserRoleMapper.insertUserRoles(vo.getId(), roleIds);
            return ResultModel.success(result);
        }
        return ResultModel.success(0);
    }

    @Operation(summary = "批量取消授权角色", description = "id为用户id,ids为角色id")
    @PostMapping("/revokeRole")
    public ResultModel<Integer> revokeRoleFromUser(@RequestBody @Valid GrantVo vo) {
        val result = sysUserRoleMapper.deleteByUserRoles(vo.getId(), vo.getTargetIds());
        return ResultModel.success(result);
    }

}
