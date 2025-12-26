package demo.controller;

import demo.entity.SysMenu;
import demo.model.RequestModel;
import demo.model.ResultModel;
import demo.model.auth.MenuSelect;
import demo.model.auth.SysMenuQuery;
import demo.repository.SysMenuRepository;
import demo.repository.SysRoleMenuRepository;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import lombok.RequiredArgsConstructor;
import lombok.val;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/// @author bin
/// @since 2023/05/30
@Tag(name = "系统菜单维护", description = "系统菜单维护相关接口")
@RestController
@RequestMapping("/sys/menu")
@RequiredArgsConstructor
public class SysMenuController {
    private final SysMenuRepository sysMenuRepository;
    private final SysRoleMenuRepository sysRoleMenuRepository;

    @Operation(summary = "系统菜单列表")
    @PostMapping("/list")
    public ResultModel<List<SysMenu>> listSysRole(@RequestBody @NotNull @Valid SysMenuQuery vo) {
        return ResultModel.success(sysMenuRepository.findAll(SysMenuQuery.orderByAsc.and(vo), vo.toPage()));
    }

    @Operation(summary = "系统菜单树")
    @PostMapping("/listTree")
    public ResultModel<List<MenuSelect>> listTree(@RequestBody @NotNull @Valid SysMenuQuery vo) {
        val list = sysMenuRepository.findAll(SysMenuQuery.orderByAsc.and(vo));
        val build = MenuSelect.build(list);
        return ResultModel.success(build);
    }

    @Operation(summary = "新增系统菜单")
    @PostMapping("/new")
    public ResultModel<?> newSysRole(@RequestBody @NotNull @Valid SysMenu sysMenu) {
        if (!sysMenuRepository.checkMenuNameUnique(sysMenu)) {
            return ResultModel.fail("新增菜单'" + sysMenu.getMenuName() + "'失败，菜单名称已存在");
        }
        sysMenuRepository.save(sysMenu);
        return ResultModel.success();
    }

    @Operation(summary = "修改系统菜单")
    @PostMapping("/update")
    public ResultModel<?> updateSysRole(@RequestBody @NotNull @Valid SysMenu sysMenu) {
        sysMenuRepository.save(sysMenu);
        return ResultModel.success();
    }

    @Operation(summary = "删除系统菜单")
    @PostMapping("/delete")
    public ResultModel<?> deleteSysRole(@RequestBody @NotNull @Valid RequestModel<List<Long>> model) {
        val ids = model.getData();
        sysRoleMenuRepository.deleteByMenuIds(ids);
        sysMenuRepository.deleteAllById(ids);
        return ResultModel.success();
    }
}
