"""Frontier mapping task; no driving before speech, no completion on blockage."""
import time
from core import frontier_plan, valid_name, amcl_map_pixels


ACTIVE = ('announcing', 'running', 'checking', 'finishing')


class Exploration:
    def __init__(self, console, maps):
        self.console = console
        self.maps = maps
        self.phase = 'idle'; self.note = ''; self.saved_map = None
        self.name = None; self.auto_save = True; self.goals = 0; self.failures = 0
        self.excluded = []; self.last_check = 0.; self.last_version = -1
        self.empty_since = None; self.empty_samples = 0; self.known = None
        self.stable_grid = None; self.stable_info = None
        self.finish_due = 0.; self.last_frontiers = 0

    def active(self): return self.phase in ACTIVE

    def status(self):
        return {'phase':self.phase,'note':self.note,'saved_map':self.saved_map,
                'map_name':self.name,'auto_save':self.auto_save,'goals':self.goals,
                'failed_goals':self.failures,'frontiers':self.last_frontiers}

    def cancel(self):
        if self.active(): self.phase='stopped';self.note='自主探索建图已停止'

    def start(self, data):
        c = self.console
        if self.active() or c.exploring: raise ValueError('自主探索建图已在运行')
        if c.search.active() or c.manual or c.goal_handle or c.pending or c.armed:
            raise ValueError('请先停止当前运动任务或键盘控制')
        if c.mode != 'mapping' or not c.ready():
            raise ValueError('请先启动 Cartographer 建图，等待地图和导航就绪')
        if not c.speaker.available(): raise ValueError('小车语音播报不可用')
        if c.speaker.busy(): raise ValueError('请等待当前播报结束')
        auto_save = data.get('auto_save',True)
        if type(auto_save) is not bool: raise ValueError('自动保存选项无效')
        name = valid_name(data.get('name') or time.strftime('explore_%Y%m%d_%H%M%S'))
        if auto_save and (self.maps/name).exists(): raise ValueError('地图名称已存在，请使用新名称')
        c.speaker.say('自主探索建图开始')
        self.__init__(c,self.maps)
        self.name=name;self.auto_save=auto_save;self.phase='announcing'
        c.exploring=True;c.armed=False;c.owner=data.get('_client');c.lease=time.monotonic()
        self.note=c.message='正在播报：自主探索建图开始；播报结束后开始移动'
        c.next_explore=0.

    def end(self, phase, note, spoken=None):
        c=self.console
        c.stop()
        self.phase=phase;self.note=c.message=note
        if spoken:
            try:c.speaker.say(spoken)
            except Exception as exc:self.note=c.message=note+'；播报失败：'+str(exc)

    def finished(self, status, goal):
        c=self.console
        if goal:self.excluded.append(goal[:2])
        if status == 4:self.goals+=1
        else:self.failures+=1
        self.empty_since=None;self.empty_samples=0
        self.last_check=time.monotonic()+3
        self.note=c.message=('正在更新地图并选择下一个探索目标' if status==4 else
                             '该探索目标无法到达，正在尝试其他未知区域')

    def tick(self):
        if not self.active():return
        c=self.console;now=time.monotonic()
        if now-c.lease>3 or not c.ready():
            return self.end('failed','自主探索已停止：页面断连或地图、传感器、导航未就绪')
        if self.phase=='announcing':
            speech=c.speaker.status()
            if speech['busy']:return
            if speech['error']:return self.end('failed','开始播报失败，未启动运动：'+speech['error'])
            self.phase='running';c.armed=True
        if self.phase=='finishing':
            if now-self.finish_due>12:
                return self.end('failed','探索已停止，但底盘尚未确认静止；请检查后手动保存地图')
            if now-self.finish_due<2 or abs(c.odom_linear)>.015 or abs(c.odom_angular)>.03:return
            if now-c.map_time>=5:return
            if frontier_plan(c.grid,c.info,c.pose,self.excluded)['reason']!='no_frontiers':
                self.phase='running';self.empty_since=None;self.empty_samples=0;self.last_check=0.
                self.note=c.message='停车复核发现新的探索边界，继续更新地图'
                return
            c.stop()
            note='自主探索建图完成：当前可达区域未发现新的可探索边界'
            if self.auto_save:
                try:self.saved_map=c.save_map(self.name)['name'];note+='；地图已保存：'+self.saved_map
                except Exception as exc:
                    self.phase='save_failed';self.note=c.message='探索结束，但自动保存失败：'+str(exc)+'；请手动保存'
                    try:c.speaker.say('探索结束，地图保存失败，请手动保存')
                    except Exception:pass
                    return
            self.phase='complete';self.note=c.message=note
            try:c.speaker.say('自主探索建图完成，地图已保存' if self.auto_save else '自主探索建图完成，请保存地图')
            except Exception as exc:self.note=c.message=note+'；播报失败：'+str(exc)
            return
        if c.goal_handle or c.pending:
            if now-c.goal_time>90:
                return self.end('failed','探索目标超时，已停止；当前地图可手动保存','自主探索已停止，请检查通道')
            return
        if now<self.last_check:return
        self.last_check=now+2
        if c.map_version==self.last_version:return
        self.last_version=c.map_version
        plan=frontier_plan(c.grid,c.info,c.pose,self.excluded)
        self.last_frontiers=plan['frontiers']
        if plan['goal']:
            self.phase='running';self.empty_since=None;self.empty_samples=0
            try:c.send_goal(*plan['goal'])
            except Exception as exc:return self.end('failed','无法发送探索目标：'+str(exc))
            self.note=c.message='正在自主探索建图：前往未知区域边界'
            return
        if plan['reason']!='no_frontiers':
            return self.end('blocked','探索已暂停：仍有未知边界，但没有可安全到达的新观察点；请检查窄门或障碍物，可手动补图后重新探索',
                            '自主探索受阻，请检查通道')
        # A failed route may hide an unobserved area; never report it as complete.
        if self.failures:
            return self.end('blocked','探索已停止：本轮存在未到达的探索目标，请检查后补图；当前地图可手动保存',
                            '自主探索结束，部分区域未完成，请检查')
        known=int((c.grid>=0).sum())
        classes=amcl_map_pixels(c.grid)
        geometry=(c.info['width'],c.info['height'],c.info['resolution'],tuple(c.info['origin']))
        changed=(self.stable_grid is None or geometry!=self.stable_info or
                 int((classes!=self.stable_grid).sum())>12)
        if self.empty_since is None or self.known is None or abs(known-self.known)>12 or changed:
            self.empty_since=now;self.empty_samples=0;self.known=known
            self.stable_grid=classes;self.stable_info=geometry
        self.empty_samples+=1;self.phase='checking'
        self.note=c.message='正在复核建图范围：等待地图稳定，确认没有新的可达探索边界'
        if now-self.empty_since>=12 and self.empty_samples>=4:
            self.phase='finishing';self.finish_due=now;c.armed=False
            self.note=c.message='可达区域探索结束，正在停车并保存地图' if self.auto_save else '可达区域探索结束，正在停车'
