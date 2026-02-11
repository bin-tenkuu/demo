package demo.auth.mapper;


import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import demo.auth.entity.SysMenu;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/// 菜单权限表 Mapper 接口
///
/// @author bin
/// @since 2023-05-30 09:40:18
public interface SysMenuMapper extends BaseMapper<SysMenu> {

    List<SysMenu> listByMenuName(String menuName);

    List<SysMenu> listMenuByUserId(@Param("userId") Long userId);
    List<SysMenu> listMenu();
}
