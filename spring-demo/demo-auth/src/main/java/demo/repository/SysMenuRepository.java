package demo.repository;


import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import demo.constant.UserConstants;
import demo.entity.SysMenu;
import demo.mapper.SysMenuMapper;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * <p>
 * 菜单权限表 服务实现类
 * </p>
 *
 * @author bin
 * @since 2023-05-30 09:40:18
 */
@Service
public class SysMenuRepository extends ServiceImpl<SysMenuMapper, SysMenu> {

    /**
     * 校验菜单名称是否唯一
     *
     * @param menu 菜单信息
     * @return 结果
     */
    public boolean checkMenuNameUnique(SysMenu menu) {
        var menus = baseMapper.listByMenuName(menu.getMenuName());
        var id = menu.getId();
        for (var sysMenu : menus) {
            if (!sysMenu.getId().equals(id)) {
                return true;
            }
        }
        return false;
    }

    public List<SysMenu> listMenuByUserId(Long userId) {
        if (userId == null || userId.equals(UserConstants.ADMIN_ID)) {
            // 管理员拥有所有权限
            return baseMapper.listMenu();
        }
        return baseMapper.listMenuByUserId(userId);
    }
}
