package demo.controller;

import demo.entity.*;
import demo.model.RequestModel;
import demo.model.ResultModel;
import demo.model.auth.GrantVo;
import demo.model.auth.SysUserAuthUpdate;
import demo.model.auth.SysUserNewVo;
import demo.model.auth.SysUserQuery;
import demo.repository.SysMenuRepository;
import demo.repository.SysUserAuthRepository;
import demo.repository.SysUserRepository;
import demo.repository.SysUserRoleRepository;
import demo.service.auth.TokenService;
import demo.util.SecurityUtils;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.persistence.criteria.JoinType;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.web.bind.annotation.*;

import java.util.HashSet;
import java.util.List;

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
    private final SysUserRoleRepository sysUserRoleRepository;

    @Operation(summary = "查询系统用户列表")
    @PostMapping("/list")
    public ResultModel<List<SysUser>> list(@RequestBody SysUserQuery query) {
        var sysUserPage = sysUserRepository.findAll(query, query.toPage());
        return ResultModel.success(sysUserPage);
    }

    @Operation(summary = "查询系统用户列表-根据角色筛选")
    @PostMapping("/list/{roleId}")
    public ResultModel<List<SysUser>> listSysUserWithRole(
            @RequestBody @Valid SysUserQuery vo,
            @PathVariable Long roleId
    ) {
        Specification<SysUser> join = (root, query, cb) -> {
            var roleMenu = root.join("userRoles", JoinType.INNER);
            return cb.equal(roleMenu.get("id").get("roleId"), roleId);
        };
        return ResultModel.success(sysUserRepository.findAll(join.and(vo), vo.toPage()));
    }

    @Operation(summary = "查询系统用户列表-未分配角色")
    @PostMapping("/list/unallocated")
    public ResultModel<List<SysUser>> unallocatedList(@RequestBody @Valid SysUserQuery vo) {
        Specification<SysUser> join = (root, query, cb) -> {
            return cb.isEmpty(root.get("userRoles"));
        };
        return ResultModel.success(sysUserRepository.findAll(join.and(vo), vo.toPage()));
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
        if (sysUserAuthRepository.checkUserNameExist(username)) {
            return ResultModel.fail("新增用户'" + username + "'失败，账号已存在");
        }
        sysUserRepository.save(sysUser);
        var id = sysUser.getId();
        var encrypt = tokenService.encode(password);
        var sysUserAuth = new SysUserAuth();
        sysUserAuth.setUsername(username);
        sysUserAuth.setPassword(encrypt);
        sysUserAuth.setType(SysUserAuthType.STATIC);
        sysUserAuth.setUserId(id);
        sysUserAuthRepository.save(sysUserAuth);
        var roleIds = vo.getRoleIds();
        if (roleIds != null && !roleIds.isEmpty()) {
            var stream = roleIds.stream()
                    .map(roleId -> new SysUserRole(id, roleId))
                    .toList();
            sysUserRoleRepository.saveAll(stream);
        }
        return ResultModel.success();
    }

    @Operation(summary = "修改系统用户")
    @PostMapping("/update")
    public ResultModel<?> updateSysUser(@RequestBody @Valid SysUser sysUser) {
        sysUserRepository.save(sysUser);
        return ResultModel.success();
    }

    @Operation(summary = "更新密码")
    @PostMapping("/updatePassword")
    public ResultModel<?> updatePassword(@RequestBody @Valid SysUserAuthUpdate update) {
        var auth = sysUserAuthRepository.findAllByUsername(update.getUsername());
        if (auth == null) {
            return ResultModel.fail("用户不存在");
        }
        if (!update.getType().equals(auth.getType())) {
            return ResultModel.fail("密码类型不匹配");
        }
        if (!SecurityUtils.getUserId().map(auth.getUserId()::equals).orElse(false)) {
            return ResultModel.fail("只能修改自己的密码");
        }
        if (tokenService.notMatches(update.getOldPassword(), auth.getPassword())) {
            return ResultModel.fail("旧密码不正确");
        }
        var encode = tokenService.encode(update.getNewPassword());
        auth.setPassword(encode);
        sysUserAuthRepository.save(auth);
        return ResultModel.success();
    }

    @Operation(summary = "删除系统用户")
    @PostMapping("/delete")
    public ResultModel<?> deleteSysUser(@RequestBody @Valid RequestModel<List<Long>> model) {
        var ids = new HashSet<>(model.getData());
        if (ids.contains(0L)) {
            return ResultModel.fail("超级管理员不能删除");
        }
        var list = sysUserRepository.findIdByStatusAndIdIn(1, ids);
        if (!list.isEmpty()) {
            // 已经停用的用户可以直接删除
            sysUserRoleRepository.deleteByUserIdIn(list);
            list.forEach(ids::remove);
            sysUserRepository.deleteAllById(list);
        }
        if (!ids.isEmpty()) {
            // 批量停用
            sysUserRepository.updateAllByIdIn(SecurityUtils.getUsername().orElse(null), ids);
        }
        return ResultModel.success();
    }

    @Operation(summary = "根据当前用户编号获取详细信息")
    @GetMapping(value = "/info")
    public ResultModel<SysUser> getInfo() {
        Long userId = SecurityUtils.getUserId().orElseThrow(() -> new RuntimeException("未登陆用户"));
        SysUser sysUser = sysUserRepository.findById(userId).orElse(null);
        return ResultModel.success(sysUser);
    }

    @Operation(summary = "根据当前用户获取菜单")
    @PostMapping("/menus")
    public ResultModel<List<SysMenu>> listMenusByUserId() {
        Long userId = SecurityUtils.getUserId().orElseThrow(() -> new RuntimeException("未登陆用户"));
        var menus = sysMenuRepository.listMenuByUserId(userId);
        return ResultModel.success(menus);
    }

    @Operation(summary = "批量授权角色", description = "id为用户id,ids为角色id")
    @PostMapping("/grantRole")
    public ResultModel<?> grantRoleToUser(@RequestBody @Valid GrantVo vo) {
        sysUserRoleRepository.deleteByUserId(vo.getId());
        var roleIds = vo.getTargetIds();
        if (roleIds != null && !roleIds.isEmpty()) {
            var list = roleIds.stream()
                    .map(roleId -> new SysUserRole(vo.getId(), roleId))
                    .toList();
            sysUserRoleRepository.saveAll(list);
        }
        return ResultModel.success();
    }

    @Operation(summary = "批量取消授权角色", description = "id为用户id,ids为角色id")
    @PostMapping("/revokeRole")
    public ResultModel<?> revokeRoleFromUser(@RequestBody @Valid GrantVo vo) {
        sysUserRoleRepository.deleteByUserIdAndRoleIdIn(vo.getId(), vo.getTargetIds());
        return ResultModel.success();
    }

}
